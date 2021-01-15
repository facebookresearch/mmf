# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Any, List

from mmf.common.registry import registry
from mmf.common.sample import SampleList
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.base import Callback


logger = logging.getLogger(__name__)


class LightningLoopCallback(Callback):
    def __init__(self, lightning_trainer: Any):
        super().__init__()
        self.lightning_trainer = lightning_trainer

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        registry.register("current_epoch", trainer.current_epoch)

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

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        # TODO: @sash next PR
        pass

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
