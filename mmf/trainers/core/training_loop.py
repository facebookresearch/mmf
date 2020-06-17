# Copyright (c) Facebook, Inc. and its affiliates.

import gc
import math
from abc import ABC
from typing import Any, Dict

import torch
from torch import Tensor

from mmf.common.registry import registry
from mmf.common.report import Report
from mmf.utils.distributed import get_world_size
from mmf.utils.general import clip_gradients


class TrainerTrainingLoopMixin(ABC):
    current_epoch: int = 0
    current_iteration: int = 0
    num_updates: int = 0

    def training_loop(self) -> None:
        self.max_updates = self.training_config.max_updates
        self.max_epochs = self.training_config.max_epochs
        if self.max_epochs is None:
            self.max_epochs = math.inf
        else:
            self.max_updates = math.inf

        torch.autograd.set_detect_anomaly(True)

        self.writer.write("Starting training...")
        self.model.train()
        self.run_training_epoch()
        self.after_training_loop()

    def after_training_loop(self) -> None:
        self.writer.write("Stepping into final validation check")
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

            for batch in self.train_loader:
                self.profile("Batch load time")
                self.current_iteration += 1
                self.writer.write(self.num_updates + 1, "debug")

                self.run_training_batch(batch)

                # Check if training should be stopped
                should_break = False

                if self.num_updates % self.training_config.evaluation_interval == 0:
                    # Validation begin callbacks
                    self.on_validation_start()

                    self.writer.write(
                        "Evaluation time. Running on full validation set..."
                    )
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
                        self.writer.write("Early stopping activated")
                        should_break = True
                if self.num_updates > self.max_updates:
                    should_break = True
                if should_break:
                    break

            # In distributed, each worker will complete one epoch when we reach this
            # as each worker is an individual instance
            self.current_epoch += get_world_size() - 1

    def run_training_batch(self, batch: Tensor) -> None:
        # Train batch start callbacks
        self.on_batch_start()

        report = self._forward(batch)
        loss = self._extract_loss(report)
        self._backward(loss)

        if self.num_updates % self.logistics_callback.log_interval == 0:
            should_log = True
            # Calculate metrics every log interval for debugging
            if self.training_config.evaluate_metrics:
                report.metrics = self.metrics(report, report)
            self.update_meter(report, self.meter)

        # Train batch end callbacks
        self.on_batch_end(report=report, meter=self.meter, should_log=should_log)

    def _forward(self, batch: Tensor) -> Dict[str, Any]:
        prepared_batch = self.dataset_loader.prepare_batch(batch)
        self.profile("Batch prepare time")
        # Arguments should be a dict at this point
        model_output = self.model(prepared_batch)
        report = Report(prepared_batch, model_output)
        self.profile("Forward time")

        return report

    def _backward(self, loss: Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()

        if self.training_config.clip_gradients:
            clip_gradients(self.model, self.num_updates, self.tb_writer, self.config)

        self.optimizer.step()
        self.num_updates += 1
        self.profile("Backward time")

    def _extract_loss(self, report: Dict[str, Any]) -> Tensor:
        loss_dict = report.losses
        loss = sum([loss.mean() for loss in loss_dict.values()])
        return loss
