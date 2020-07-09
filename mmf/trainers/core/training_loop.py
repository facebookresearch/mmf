# Copyright (c) Facebook, Inc. and its affiliates.

import gc
import logging
import warnings
from abc import ABC
from typing import Any, Dict

import torch
from mmf.common.registry import registry
from mmf.common.report import Report
from mmf.common.sample import to_device
from mmf.utils.general import assert_iterator_finished, clip_gradients
from torch import Tensor


logger = logging.getLogger(__name__)


class TrainerTrainingLoopMixin(ABC):
    current_epoch: int = 0
    current_iteration: int = 0
    num_updates: int = 0

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
            and self.num_updates % self.training_config.evaluation_interval != 0
        ):
            # Create a new meter for this case
            report, meter = self.evaluation_loop(self.val_loader)

            # Validation end callbacks
            self.on_validation_end(report=report, meter=meter)

    def run_training_epoch(self) -> None:
        should_break = False
        while self.num_updates < self.max_updates and not should_break:
            self.current_epoch += 1
            registry.register("current_epoch", self.current_epoch)

            # Seed the sampler in case if it is distributed
            self.dataset_loader.seed_sampler("train", self.current_epoch)

            if self.current_epoch > self.max_epochs:
                break

            num_remaining_batches = len(self.train_loader)
            batch_iter = iter(self.train_loader)

            while num_remaining_batches:

                combined_report = None
                num_batches_for_this_update = min(
                    self.config.training.update_frequency, num_remaining_batches
                )

                self._start_update()

                for _ in range(num_batches_for_this_update):
                    batch = next(batch_iter)
                    self.profile("Batch load time")

                    report = self.run_training_batch(batch, num_batches_for_this_update)

                    # accumulate necessary params for metric calculation
                    if combined_report is None:
                        combined_report = report
                    else:
                        combined_report.accumulate_tensor_fields(
                            report, self.metrics.required_params
                        )
                        combined_report.batch_size += report.batch_size

                self._finish_update()

                should_log = False
                if self.num_updates % self.logistics_callback.log_interval == 0:
                    should_log = True
                    # Calculate metrics every log interval for debugging
                    if self.training_config.evaluate_metrics:
                        combined_report.metrics = self.metrics(
                            combined_report, combined_report
                        )
                    self.update_meter(combined_report, self.meter)

                self.on_update_end(
                    report=combined_report, meter=self.meter, should_log=should_log
                )

                num_remaining_batches -= num_batches_for_this_update

                if num_remaining_batches == 0:
                    # we expect iterator to be finished, based on len(self.train_loader)
                    assert_iterator_finished(batch_iter)

                # Check if training should be stopped
                should_break = False

                if self.num_updates % self.training_config.evaluation_interval == 0:
                    # Validation begin callbacks
                    self.on_validation_start()

                    logger.info("Evaluation time. Running on full validation set...")
                    # Validation and Early stopping
                    # Create a new meter for this case
                    report, meter = self.evaluation_loop(self.val_loader)

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

    def run_training_batch(self, batch: Tensor, loss_divisor: int) -> None:

        report = self._forward(batch)
        loss = self._extract_loss(report)
        loss /= loss_divisor
        self._backward(loss)

        return report

    def _forward(self, batch: Tensor) -> Dict[str, Any]:
        prepared_batch = self.dataset_loader.prepare_batch(batch)
        # Move the sample list to device if it isn't as of now.
        prepared_batch = to_device(prepared_batch, torch.device("cuda"))
        self.profile("Batch prepare time")
        # Arguments should be a dict at this point
        model_output = self.model(prepared_batch)
        report = Report(prepared_batch, model_output)
        self.profile("Forward time")

        return report

    def _start_update(self):
        self.current_iteration += 1
        logger.debug(self.num_updates + 1)
        self.optimizer.zero_grad()

    def _backward(self, loss: Tensor) -> None:
        loss.backward()
        self.profile("Backward time")

    def _finish_update(self):
        if self.training_config.clip_gradients:
            clip_gradients(
                self.model,
                self.num_updates,
                self.logistics_callback.tb_writer,
                self.config,
            )

        self.optimizer.step()
        self.num_updates += 1
        self.profile("Finish update")

    def _extract_loss(self, report: Dict[str, Any]) -> Tensor:
        loss_dict = report.losses
        loss = sum([loss.mean() for loss in loss_dict.values()])
        return loss

    def _calculate_max_updates(self):
        max_updates = self.training_config.max_updates
        max_epochs = self.training_config.max_epochs
        if max_updates is None and max_epochs is None:
            raise ValueError("Neither max_updates nor max_epochs is specified.")

        if max_updates is not None and max_epochs is not None:
            warnings.warn(
                "Both max_updates and max_epochs are specified. "
                + f"Favoring max_epochs: {max_epochs}"
            )

        if max_epochs is not None:
            max_updates = len(self.train_loader) * max_epochs

        return max_updates
