# Copyright (c) Facebook, Inc. and its affiliates.

import gc
import logging
from abc import ABC
from typing import Any, Dict

import torch
from mmf.common.meter import Meter
from mmf.common.registry import registry
from mmf.common.report import Report
from mmf.common.sample import to_device
from mmf.utils.distributed import is_xla
from mmf.utils.general import clip_gradients, extract_loss, get_max_updates
from torch import Tensor


logger = logging.getLogger(__name__)


class TrainerTrainingLoopMixin(ABC):
    current_epoch: int = 0
    current_iteration: int = 0
    num_updates: int = 0
    meter: Meter = Meter()

    def training_loop(self) -> None:
        self.max_updates = self._calculate_max_updates()
        torch.autograd.set_detect_anomaly(self.training_config.detect_anomaly)

        logger.info("Starting training...")
        self.model.train()
        self.run_training_epoch()
        self.after_training_loop()

    def after_training_loop(self) -> None:
        logger.info("Stepping into final validation check")
        # Only do when run_type has train as it shouldn't happen on validation and
        # inference runs. Inference will take care of this anyways. Also, don't run
        # if current iteration is divisble by snapshot interval as it will just
        # be a repeat
        if (
            "train" in self.run_type
            and "val" in self.run_type
            and self.num_updates % self.training_config.evaluation_interval != 0
        ):
            # Create a new meter for this case
            report, meter = self.evaluation_loop("val")

            # Validation end callbacks
            self.on_validation_end(report=report, meter=meter)

    def run_training_epoch(self) -> None:
        should_break = False
        while self.num_updates < self.max_updates and not should_break:
            self.current_epoch += 1
            registry.register("current_epoch", self.current_epoch)

            # Seed the sampler in case if it is distributed
            self.dataset_loader.seed_sampler("train", self.current_epoch)

            # For iterable datasets we cannot determine length of dataset properly.
            # For those cases we set num_remaining_batches to be the (number of
            # updates remaining x update_frequency)
            num_remaining_batches = (
                (
                    (self.max_updates - self.num_updates)
                    * self.training_config.update_frequency
                )
                if isinstance(
                    self.train_loader.current_dataset, torch.utils.data.IterableDataset
                )
                else len(self.train_loader)
            )

            should_start_update = True
            for idx, batch in enumerate(self.train_loader):
                if should_start_update:
                    combined_report = None
                    self._start_update()
                    num_batches_for_this_update = min(
                        self.training_config.update_frequency, num_remaining_batches
                    )
                    should_start_update = False

                self.current_iteration += 1
                # batch execution starts here
                self.on_batch_start()
                self.profile("Batch load time")

                report = self.run_training_batch(batch, num_batches_for_this_update)
                report = report.detach()

                # accumulate necessary params (including loss) for metric calculation
                if combined_report is None:
                    combined_report = report
                else:
                    combined_report.accumulate_tensor_fields_and_loss(
                        report, self.metrics.required_params
                    )
                    combined_report.batch_size += report.batch_size

                # batch execution ends here
                self.on_batch_end(report=combined_report, meter=self.meter)

                # check if an update has finished or if it is the last, if no continue
                if (
                    (idx + 1) % self.training_config.update_frequency
                    and num_remaining_batches != num_batches_for_this_update
                ):
                    continue

                self._finish_update()
                should_start_update = True

                should_log = False
                if self.num_updates % self.logistics_callback.log_interval == 0:
                    should_log = True
                    # Calculate metrics every log interval for debugging
                    if self.training_config.evaluate_metrics:
                        combined_report.metrics = self.metrics(
                            combined_report, combined_report
                        )
                    self.meter.update_from_report(combined_report)

                self.on_update_end(
                    report=combined_report, meter=self.meter, should_log=should_log
                )

                num_remaining_batches -= num_batches_for_this_update

                # Check if training should be stopped
                should_break = False

                if self.num_updates % self.training_config.evaluation_interval == 0:
                    # Validation begin callbacks
                    self.on_validation_start()

                    logger.info("Evaluation time. Running on full validation set...")
                    # Validation and Early stopping
                    # Create a new meter for this case
                    report, meter = self.evaluation_loop("val")

                    # Validation end callbacks
                    stop = self.early_stop_callback.on_validation_end(
                        report=report, meter=meter
                    )
                    self.on_validation_end(report=report, meter=meter)

                    gc.collect()

                    if "cuda" in str(self.device):
                        torch.cuda.empty_cache()

                    if stop is True:
                        logger.info("Early stopping activated")
                        should_break = True

                if self.num_updates >= self.max_updates:
                    should_break = True

                if should_break:
                    break

    def run_training_batch(self, batch: Dict[str, Tensor], loss_divisor: int) -> None:
        report = self._forward(batch)
        if self.training_config.exit_on_nan_losses:
            self._check_nan_losses(report)
        loss = extract_loss(report, loss_divisor)
        self._backward(loss)
        return report

    def _check_nan_losses(self, report):
        # skip this check in XLA mode as calling .item() in forward pass
        # greatly slows down the training
        if not is_xla():
            # check whether NaN has occurred in the losses, and exit the training
            # when NaN happens
            loss_dict = report.losses
            nan_loss_keys = []
            for key, value in loss_dict.items():
                if torch.any(torch.isnan(value)).item():
                    nan_loss_keys.append(key)
            if len(nan_loss_keys) > 0:
                keys_str = ", ".join(nan_loss_keys)
                error_msg = (
                    f"NaN occurred in the following loss(es): {keys_str}; "
                    f"exiting the training"
                )
                logger.info(error_msg)
                raise RuntimeError(error_msg)

    def _forward(self, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        # Move the sample list to device if it isn't as of now.
        prepared_batch = to_device(batch, self.device)
        self.profile("Batch prepare time")
        # Arguments should be a dict at this point

        with torch.cuda.amp.autocast(enabled=self.training_config.fp16):
            model_output = self.model(prepared_batch)
            report = Report(prepared_batch, model_output)

        self.profile("Forward time")
        return report

    def _start_update(self):
        logger.debug(self.num_updates + 1)
        self.on_update_start()
        self.optimizer.zero_grad()

    def _backward(self, loss: Tensor) -> None:
        self.scaler.scale(loss).backward()
        self.profile("Backward time")

    def _finish_update(self):
        if self.training_config.clip_gradients:
            clip_gradients(
                self.model,
                self.optimizer,
                self.num_updates,
                self.logistics_callback.tb_writer,
                self.config,
                scale=self.scaler.get_scale(),
            )
        if is_xla():
            import torch_xla.core.xla_model as xm

            # Assumes no model parallel
            xm.reduce_gradients(self.optimizer)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.num_updates += 1
        self.profile("Finished update")

    def _calculate_max_updates(self):
        config_max_updates = self.training_config.max_updates
        config_max_epochs = self.training_config.max_epochs
        max_updates, _ = get_max_updates(
            config_max_updates,
            config_max_epochs,
            self.train_loader,
            self.training_config.update_frequency,
        )

        return max_updates
