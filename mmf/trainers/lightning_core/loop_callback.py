# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Any, Dict, List

import torch
from mmf.common.meter import Meter
from mmf.common.registry import registry
from mmf.common.report import Report
from mmf.common.sample import SampleList
from mmf.utils.logger import calculate_time_left, summarize_report
from mmf.utils.timer import Timer
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.base import Callback


logger = logging.getLogger(__name__)


class LightningLoopCallback(Callback):
    def __init__(self, lightning_trainer: Any):
        super().__init__()
        self.lightning_trainer = lightning_trainer
        # this is lightning trainer's config
        self.trainer_config = lightning_trainer.trainer_config
        # training config configures training parameters.
        self.training_config = lightning_trainer.training_config
        self.run_type = lightning_trainer.run_type

        # for logging
        self.total_timer = Timer()
        self.snapshot_timer = Timer()
        self.snapshot_iterations = 0
        if self.lightning_trainer.val_loader.has_len():
            self.snapshot_iterations = len(self.lightning_trainer.val_loader)
        self.train_timer = Timer()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        registry.register("current_epoch", trainer.current_epoch)
        self.train_combined_report = None

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
        self.train_combined_report = self._update_and_create_report(
            SampleList(batch), batch_idx, outputs, pl_module, self.train_combined_report
        )

        # Continue if an update has not finished
        if (batch_idx + 1) % self.trainer_config.accumulate_grad_batches:
            return

        # log
        if (
            self._get_num_updates_for_logging(trainer)
            % self.trainer_config.log_every_n_steps
            == 0
        ):
            self._train_log(trainer, pl_module)

    # Validation Callbacks
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule):
        logger.info("Evaluation time. Running on full validation set...")
        self.snapshot_timer.reset()
        self.val_combined_report = None
        pl_module.val_meter.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: SampleList,
        batch_idx: int,
        dataloader_idx: int,
    ):
        # prepare the next batch
        self.lightning_trainer.data_module.val_loader.change_dataloader()

        # aggregate val_combined_report
        self.val_combined_report = self._update_and_create_report(
            batch,
            batch_idx,
            outputs,
            pl_module,
            self.val_combined_report,
            update_meter=pl_module.val_meter,
        )
        self.val_combined_report = self.val_combined_report.detach()
        self.val_combined_report.metrics = pl_module.metrics(
            self.val_combined_report, self.val_combined_report
        )
        pl_module.val_meter.update_from_report(
            self.val_combined_report, should_update_loss=False
        )

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
            meter=pl_module.val_meter,
            extra=extra,
            tb_writer=self.lightning_trainer.tb_writer,
        )

    def _update_and_create_report(
        self,
        batch: Dict,
        batch_idx: int,
        step_output: Dict,
        pl_module: LightningModule,
        combined_report: Report = None,
        update_meter: Meter = None,
    ):
        report = Report(batch, step_output)

        # Normalize losses
        for key in report.losses.keys():
            report.losses[key] = (
                report.losses[key] / self.trainer_config.accumulate_grad_batches
            )

        if update_meter:
            update_meter.update_from_report(report)

        should_accumulate = not (
            batch_idx % self.trainer_config.accumulate_grad_batches == 0
        )

        final_report = report
        if should_accumulate and combined_report is not None:
            combined_report.accumulate_tensor_fields_and_loss(
                report, pl_module.metrics.required_params
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

    def _get_current_epoch_for_logging(self, trainer: Trainer):
        return trainer.current_epoch + 1

    def _get_iterations_for_logging(self, trainer: Trainer):
        return trainer.fit_loop.batch_idx + 1

    def _get_num_updates_for_logging(self, trainer: Trainer):
        return trainer.global_step

    def _train_log(self, trainer: Trainer, pl_module: LightningModule):
        self.train_combined_report = self.train_combined_report.detach()
        if self.training_config.evaluate_metrics:
            self.train_combined_report.metrics = pl_module.metrics(
                self.train_combined_report, self.train_combined_report
            )

        pl_module.train_meter.update_from_report(self.train_combined_report)

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
            meter=pl_module.train_meter,
            extra=extra,
            tb_writer=self.lightning_trainer.tb_writer,
        )
