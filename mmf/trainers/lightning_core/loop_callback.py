# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Any, Dict, List

import torch
from mmf.common.meter import Meter
from mmf.common.registry import registry
from mmf.common.report import Report
from mmf.common.sample import SampleList
from mmf.trainers.core.reporting import TrainerReportingMixin
from mmf.utils.logger import calculate_time_left, summarize_report
from mmf.utils.timer import Timer
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.base import Callback


logger = logging.getLogger(__name__)


class LightningLoopCallback(Callback, TrainerReportingMixin):
    def __init__(self, lightning_trainer: Any):
        super().__init__()
        self.lightning_trainer = lightning_trainer
        self.trainer_config = lightning_trainer.trainer_config
        self.training_config = lightning_trainer.training_config

        # for logging
        self.total_timer = Timer()
        self.snapshot_iterations = len(lightning_trainer.data_module.val_loader)
        self.snapshot_iterations //= self.training_config.batch_size

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        registry.register("current_epoch", trainer.current_epoch)
        self.train_combined_report = None

        self.train_timer = Timer()
        self.snapshot_timer = Timer()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: List,
        batch: SampleList,
        batch_idx: int,
        dataloader_idx: int,
    ):
        # prepare the next batch
        self.lightning_trainer.data_module.train_loader.change_dataloader()

        # aggregate train combined_report
        step_output = outputs[0][0]["extra"]
        self.train_combined_report = self._update_and_create_report(
            step_output, batch_idx, self.train_combined_report
        )

        # log
        if (trainer.global_step + 1) % self.trainer_config.log_every_n_steps == 0:
            self._train_log(trainer)

        # save checkpoints - TODO: @sash

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        # Only do when run_type has train as it shouldn't happen on validation and
        # inference runs. Inference will take care of this anyways. Also, don't run
        # if current iteration is divisble by snapshot interval as it will just
        # be a repeat
        if (
            "train" in self.lightning_trainer.run_type
            and trainer.global_step % self.trainer_config.val_check_interval != 0
        ):
            logger.info("Stepping into final validation check")
            # Pytorch Lightning upgrades PR4945 and PR4948 will be enabled in 1.2
            # TODO: perform final validation check

    # Validation Callbacks
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule):
        logger.info("Evaluation time. Running on full validation set...")
        self.snapshot_timer.reset()
        self.val_meter = Meter()
        self.val_combined_report = None

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: List,
        batch: SampleList,
        batch_idx: int,
        dataloader_idx: int,
    ):
        # prepare the next batch
        self.lightning_trainer.data_module.val_loader.change_dataloader()

        # aggregate val_combined_report
        self.val_combined_report = self._update_and_create_report(
            outputs, batch_idx, self.val_combined_report, update_meter=self.val_meter
        )
        self.val_combined_report.metrics = self.lightning_trainer.metrics(
            self.val_combined_report, self.val_combined_report
        )
        self.update_meter(self.val_combined_report, self.val_meter, eval_mode=True)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        iterations = self._get_iterations_for_logging(trainer)
        current_epochs = self._get_current_epoch_for_logging(trainer)
        num_updates = self._get_num_updates_for_logging(trainer)
        extra = {
            "num_updates": num_updates,
            "epoch": current_epochs,
            "iterations": iterations,
            "max_updates": trainer.max_steps,
            "val_time": self.snapshot_timer.get_time_since_start(),
        }
        # TODO: @sash populate early stop info for logging (next mvp)
        # extra.update(self.trainer.early_stop_callback.early_stopping.get_info())
        self.train_timer.reset()
        summarize_report(
            current_iteration=iterations,
            num_updates=num_updates,
            max_updates=trainer.max_steps,
            meter=self.val_meter,
            extra=extra,
            tb_writer=self.lightning_trainer.tb_writer,
        )

    def _update_and_create_report(
        self,
        step_output: Dict,
        batch_idx: int,
        combined_report: Report = None,
        update_meter: Meter = None,
    ):
        input_batch = step_output["input_batch"]
        report = Report(input_batch, step_output)

        if update_meter:
            self.update_meter(report, update_meter)

        should_accumulate = not (
            batch_idx % self.trainer_config.accumulate_grad_batches == 0
        )

        final_report = report
        if should_accumulate and combined_report is not None:
            combined_report.accumulate_tensor_fields_and_loss(
                report, self.lightning_trainer.metrics.required_params
            )
            combined_report.batch_size += report.batch_size
            final_report = combined_report

        return final_report

    def get_optimizer(self, trainer: Trainer):
        assert (
            len(trainer.optimizers) == 1
        ), "mmf lightning_trainer supports 1 optimizer per model for now."
        optimizer = trainer.optimizers[0]
        return optimizer

    def _save_checkpoint(self, trainer: Trainer):
        logger.info("Checkpoint time. Saving a checkpoint.")
        return
        # TODO: sash Needs implementation - next mvp

    def _get_current_epoch_for_logging(self, trainer: Trainer):
        return trainer.current_epoch + 1

    def _get_iterations_for_logging(self, trainer: Trainer):
        return trainer.batch_idx + 1

    def _get_num_updates_for_logging(self, trainer: Trainer):
        return trainer.global_step + 1

    def _train_log(self, trainer: Trainer):
        if self.training_config.evaluate_metrics:
            self.train_combined_report.metrics = self.lightning_trainer.metrics(
                self.train_combined_report, self.train_combined_report
            )
        self.update_meter(self.train_combined_report, self.meter)

        extra = {}
        if "cuda" in str(trainer.model.device):
            extra["max mem"] = torch.cuda.max_memory_allocated() / 1024
            extra["max mem"] //= 1024

        if self.training_config.experiment_name:
            extra["experiment"] = self.training_config.experiment_name

        optimizer = self.get_optimizer(trainer)
        num_updates = self._get_num_updates_for_logging(trainer)
        current_iteration = self._get_iterations_for_logging(trainer)
        extra.update(
            {
                "epoch": self._get_current_epoch_for_logging(trainer),
                "iterations": current_iteration,
                "num_updates": num_updates,
                "max_updates": trainer.max_steps,
                "lr": "{:.5f}".format(optimizer.param_groups[0]["lr"]).rstrip("0"),
                "ups": "{:.2f}".format(
                    self.trainer_config.log_every_n_steps
                    / self.train_timer.unix_time_since_start()
                ),
                "time": self.train_timer.get_time_since_start(),
                "time_since_start": self.total_timer.get_time_since_start(),
                "eta": calculate_time_left(
                    max_updates=trainer.max_steps,
                    num_updates=num_updates,
                    timer=self.train_timer,
                    num_snapshot_iterations=self.snapshot_iterations,
                    log_interval=self.trainer_config.log_every_n_steps,
                    eval_interval=self.trainer_config.val_check_interval,
                ),
            }
        )
        self.train_timer.reset()
        summarize_report(
            current_iteration=current_iteration,
            num_updates=num_updates,
            max_updates=trainer.max_steps,
            meter=self.meter,
            extra=extra,
            tb_writer=self.lightning_trainer.tb_writer,
        )
