# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock, patch

from mmf.common.report import Report
from pytorch_lightning.callbacks.base import Callback
from tests.trainers.test_utils import (
    get_config_with_defaults,
    get_lightning_trainer,
    get_mmf_trainer,
)


class TestLightningTrainerLoss(unittest.TestCase, Callback):
    def setUp(self):
        self.lightning_losses = []
        self.mmf_losses = []

    def test_loss_computation_parity_with_mmf_trainer(self):
        # compute mmf_trainer training losses
        def _on_update_end(report, meter, should_log):
            self.mmf_losses.append(report["losses"]["loss"].item())

        config = get_config_with_defaults(
            {"training": {"max_updates": 5, "max_epochs": None}}
        )
        mmf_trainer = get_mmf_trainer(config=config)
        mmf_trainer.on_update_end = _on_update_end
        mmf_trainer.evaluation_loop = MagicMock(return_value=(None, None))
        mmf_trainer.training_loop()

        # compute lightning_trainer training losses
        with patch("mmf.trainers.lightning_trainer.get_mmf_env", return_value=""):
            config = get_config_with_defaults({"trainer": {"params": {"max_steps": 5}}})
            trainer = get_lightning_trainer(config=config)
            trainer.callbacks.append(self)
            trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        report = Report(outputs["input_batch"], outputs)
        self.lightning_losses.append(report["losses"]["loss"].item())

    def on_train_end(self, trainer, pl_module):
        for lightning_loss, mmf_loss in zip(self.lightning_losses, self.mmf_losses):
            self.assertEqual(lightning_loss, mmf_loss)
