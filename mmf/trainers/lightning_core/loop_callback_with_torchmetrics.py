# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Any, Dict, List, Optional

import torch
from mmf.common.registry import registry
from mmf.common.sample import SampleList
from mmf.utils.timer import Timer
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.base import Callback

logger = logging.getLogger(__name__)


class LightningTorchMetricsCallback(Callback):
    """
    callback to be used with LightningTrainer and torchmetric
    Warning: 'optimizer.enable_state_sharding=True' is not supported
    """

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
        self.train_timer = Timer()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        registry.register("current_epoch", trainer.current_epoch)
        self.lightning_trainer.torchmetrics.reset()

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
        self.lightning_trainer.torchmetrics.update(batch, outputs)
        # log
        if (
            self._get_num_updates_for_logging(trainer)
            % self.trainer_config.log_every_n_steps
            == 0
        ):
            num_updates = self._get_num_updates_for_logging(trainer)
            max_updates = trainer.max_steps
            extra = self._get_train_extra_log(trainer, pl_module)
            self._log_metrics_and_extra(
                extra, num_updates, max_updates, log_type="train"
            )
            self.lightning_trainer.torchmetrics.reset()
            self.train_timer.reset()

    # Validation Callbacks
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule):
        logger.info("Evaluation time. Running on full validation set...")
        self.snapshot_timer.reset()
        self.lightning_trainer.torchmetrics.reset()

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
        self.lightning_trainer.torchmetrics.update(batch, outputs)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        iterations = self._get_iterations_for_logging(trainer)
        current_epochs = self._get_current_epoch_for_logging(trainer)
        num_updates = self._get_num_updates_for_logging(trainer)
        max_updates = trainer.max_steps
        extra = {
            "num_updates": num_updates,
            "epoch": current_epochs,
            "iterations": iterations,
            "max_updates": max_updates,
            "val_time": self.snapshot_timer.get_time_since_start(),
        }
        self.train_timer.reset()
        self._log_metrics_and_extra(extra, num_updates, max_updates, log_type="val")
        self.lightning_trainer.torchmetrics.reset()

    def _log_metrics_and_extra(
        self,
        extra: Optional[Dict],
        num_updates: int,
        max_updates: int,
        log_type: str = "train",
    ):
        logger.info(f"{num_updates}/{max_updates}")
        if extra is not None:
            logger.info(", ".join([f"{key}: {value}" for key, value in extra.items()]))
        scalar_dict = self.lightning_trainer.torchmetrics.get_scalar_dict()
        scalar_dict_with_type = {f"{log_type}_{k}": v for k, v in scalar_dict.items()}
        if self.lightning_trainer.tb_writer is not None:
            self.lightning_trainer.tb_writer.add_scalars(
                scalar_dict_with_type, num_updates
            )
        logger.info(f"{log_type} metrics:")
        logger.info(
            ", ".join([f"{key}: {value}" for key, value in scalar_dict.items()])
        )

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

    def _get_train_extra_log(self, trainer: Trainer, pl_module: LightningModule):
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
            }
        )
        return extra
