# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from abc import ABC
from typing import Any, Dict, Tuple, Type

import torch
import tqdm
from mmf.common.meter import Meter
from mmf.common.report import Report
from mmf.common.sample import to_device
from mmf.utils.distributed import gather_tensor, is_main, is_xla


logger = logging.getLogger(__name__)


class TrainerEvaluationLoopMixin(ABC):
    def evaluation_loop(
        self, dataset_type: str, use_tqdm: bool = False, single_batch: bool = False
    ) -> Tuple[Dict[str, Any], Type[Meter]]:
        meter = Meter()
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        use_cpu = self.config.evaluation.get("use_cpu", False)
        loaded_batches = 0
        skipped_batches = 0

        with torch.no_grad():
            self.model.eval()
            disable_tqdm = not use_tqdm or not is_main()
            while reporter.next_dataset(flush_report=False):
                dataloader = reporter.get_dataloader()
                combined_report = None

                if self._can_use_tqdm(dataloader):
                    dataloader = tqdm.tqdm(dataloader, disable=disable_tqdm)
                for batch in dataloader:
                    loaded_batches += 1
                    prepared_batch = reporter.prepare_batch(batch)
                    prepared_batch = to_device(prepared_batch, self.device)
                    if not validate_batch_sizes(prepared_batch.get_batch_size()):
                        logger.info("Skip batch due to uneven batch sizes.")
                        skipped_batches += 1
                        continue
                    model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    report = report.detach()

                    meter.update_from_report(report)

                    moved_report = report
                    # Move to CPU for metrics calculation later if needed
                    # Explicitly use `non_blocking=False` as this can cause
                    # race conditions in next accumulate
                    if use_cpu:
                        moved_report = report.copy().to("cpu", non_blocking=False)

                    # accumulate necessary params for metric calculation
                    if combined_report is None:
                        # make a copy of report since `reporter.add_to_report` will
                        # change some of the report keys later
                        combined_report = moved_report.copy()
                    else:
                        combined_report.accumulate_tensor_fields_and_loss(
                            moved_report, self.metrics.required_params
                        )
                        combined_report.batch_size += moved_report.batch_size

                    # Each node generates a separate copy of predict JSON from the
                    # report, which will be used to evaluate dataset-level metrics
                    # (such as mAP in object detection or CIDEr in image captioning)
                    # Since `reporter.add_to_report` changes report keys,
                    # (e.g scores) do this after
                    # `combined_report.accumulate_tensor_fields_and_loss`
                    if "__prediction_report__" in self.metrics.required_params:
                        # Still need to use original report here on GPU/TPU since
                        # it will be gathered
                        reporter.add_to_report(report, self.model)

                    if single_batch is True:
                        break

                logger.info(f"Finished evaluation inference. Loaded {loaded_batches}")
                logger.info(f" -- skipped {skipped_batches} batches.")

                reporter.postprocess_dataset_report()
                assert (
                    combined_report is not None
                ), "Please check if your validation set is empty!"
                # add prediction_report is used for set-level metrics
                combined_report.prediction_report = reporter.report

                combined_report.metrics = self.metrics(combined_report, combined_report)

                # Since update_meter will reduce the metrics over GPUs, we need to
                # move them back to GPU but we will only move metrics and losses
                # which are needed by update_meter to avoid OOM
                # Furthermore, do it in a non_blocking way to avoid any issues
                # in device to host or host to device transfer
                if use_cpu:
                    combined_report = combined_report.to(
                        self.device, fields=["metrics", "losses"], non_blocking=False
                    )

                meter.update_from_report(combined_report, should_update_loss=False)

            # enable train mode again
            self.model.train()

        return combined_report, meter

    def prediction_loop(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        skipped_batches = 0
        loaded_batches = 0
        with torch.no_grad():
            self.model.eval()
            logger.info(f"Starting {dataset_type} inference predictions")

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()
                if self._can_use_tqdm(dataloader):
                    dataloader = tqdm.tqdm(dataloader)
                for batch in dataloader:
                    prepared_batch = reporter.prepare_batch(batch)
                    prepared_batch = to_device(prepared_batch, self.device)
                    loaded_batches += 1
                    if not validate_batch_sizes(prepared_batch.get_batch_size()):
                        logger.info("Skip batch due to unequal batch sizes.")
                        skipped_batches += 1
                        continue
                    with torch.cuda.amp.autocast(enabled=self.training_config.fp16):
                        model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report, self.model)
                    report.detach()

                reporter.postprocess_dataset_report()

            logger.info(f"Finished predicting. Loaded {loaded_batches}")
            logger.info(f" -- skipped {skipped_batches} batches.")
            self.model.train()

    def _can_use_tqdm(self, dataloader: torch.utils.data.DataLoader):
        """
        Checks whether tqdm can be gracefully used with a dataloader
        1) should have `__len__` property defined
        2) calling len(x) should not throw errors.
        """
        use_tqdm = hasattr(dataloader, "__len__")

        try:
            _ = len(dataloader)
        except (AttributeError, TypeError, NotImplementedError):
            use_tqdm = False
        return use_tqdm


def validate_batch_sizes(my_batch_size: int) -> bool:
    """
    Validates all workers got the same batch size.
    """

    # skip batch size validation on XLA (as there's too much overhead
    # and data loader automatically drops the last batch in XLA mode)
    if is_xla():
        return True

    batch_size_tensor = torch.IntTensor([my_batch_size])
    if torch.cuda.is_available():
        batch_size_tensor = batch_size_tensor.cuda()
    all_batch_sizes = gather_tensor(batch_size_tensor)
    for j, oth_batch_size in enumerate(all_batch_sizes.data):
        if oth_batch_size != my_batch_size:
            logger.error(f"Node {j} batch {oth_batch_size} != {my_batch_size}")
            return False
    return True
