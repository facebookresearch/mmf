# Copyright (c) Facebook, Inc. and its affiliates.

import logging

from mmf.common.registry import registry
from pytorch_lightning.callbacks.base import Callback


logger = logging.getLogger(__name__)


class LightningLoopCallback(Callback):
    def __init__(self, lightning_trainer):
        super().__init__()
        self.lightning_trainer = lightning_trainer

    def on_init_start(self, trainer):
        pass

    def on_train_start(self, trainer, pl_module):
        registry.register("current_epoch", trainer.current_epoch)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # prepare the next batch
        self.lightning_trainer.data_module.train_loader.change_dataloader()

    def on_train_end(self, trainer, pl_module):
        trainer.run_evaluation(test_mode=False)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # prepare the next batch
        self.lightning_trainer.data_module.val_loader.change_dataloader()
