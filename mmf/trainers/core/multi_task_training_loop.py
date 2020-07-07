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
from mmf.trainers.core.training_loop import TrainerTrainingLoopMixin


class MultiTaskTrainerTrainingLoopMixin(TrainerTrainingLoopMixin):
    def __init__(self, config):
        super().__init__(config)

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

        # Seed the sampler in case if it is distributed
        self.dataset_loader.seed_sampler("train", self.current_epoch)
        registry.register("current_epoch", self.current_epoch)

        self.train_loader = iter(self.train_loader)

        while self.current_iteration < self.training_config.max_iterations:

            for task in self.config.multi_task_config.tasks:
                task_id = task.id
                dataset = task.dataset
                self.train_loader.set_dataset(dataset)
                batch = next(self.train_loader)
                if not self.task_stop_controller[task_id].in_stop:
                    self.profile("Batch load time")
                    self.writer.write(self.num_updates + 1, "debug")
                    report = self.run_training_batch(batch)
                else:
                    self.writer.write("Dataset: {} in plateu stop".format(dataset))

            should_log = False
            if self.current_iteration % self.logistics_callback.log_interval == 0:
                should_log = True
            # Train batch end callbacks
            self.on_batch_end(report=report, meter=self.meter, should_log=should_log)

            if (
                self.current_iteration + 1 % self.training_config.evaluation_interval
                == 0
            ):
                # Validation begin callbacks
                self.on_validation_start()

                self.writer.write("Evaluation time. Running on full validation set...")
                # Validation and Early stopping
                # Create a new meter for this case
                reports, meters, combined_meter = self.evaluation_loop(self.val_loader)

                for task in self.config.multi_task_config.tasks:
                    task_id = task.id
                    metric_name = "{}/{}/{}"
                    metric_name = metric_name.format(
                        reports[task_id].dataset_type,
                        reports[task_id].dataset_name,
                        task.metric,
                    )
                    metrics = reports[task_id].metrics[metric_name]
                    self.task_stop_controller[task_id].step(metrics)

                self.on_validation_end(report=None, meter=combined_meter)

                gc.collect()

                if "cuda" in str(self.device):
                    torch.cuda.empty_cache()

            self.current_iteration += 1

            # In distributed, each worker will complete one epoch when we reach this
            # as each worker is an individual instance
            self.current_epoch += get_world_size() - 1

    def run_training_batch(self, batch: Tensor) -> None:
        # Train batch start callbacks
        self.on_batch_start()

        report = self._forward(batch)
        loss = self._extract_loss(report)
        self._backward(loss)

        if self.current_iteration % self.logistics_callback.log_interval == 0:
            # Calculate metrics every log interval for debugging
            if self.training_config.evaluate_metrics:
                report.metrics = self.metrics(report, report)
            self.update_meter(report, self.meter)

        return report

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
