# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import patch

from tests.trainers.test_utils import get_config_with_defaults, get_lightning_trainer


class TestLightningTrainer(unittest.TestCase):
    def test_epoch_over_updates(self):
        with patch("mmf.trainers.lightning_trainer.get_mmf_env", return_value=""):
            config = self._get_config(max_steps=2, max_epochs=0.04)
            trainer = get_lightning_trainer(config=config)
            self.assertEqual(trainer._max_updates, 4)

            self._check_values(trainer, 0, 0)
            trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
            self._check_values(trainer, 4, 1)

    def test_fractional_epoch(self):
        with patch("mmf.trainers.lightning_trainer.get_mmf_env", return_value=""):
            config = self._get_config(max_steps=None, max_epochs=0.04)
            trainer = get_lightning_trainer(config=config)
            self.assertEqual(trainer._max_updates, 4)

            self._check_values(trainer, 0, 0)
            trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
            self._check_values(trainer, 4, 1)

    def test_updates(self):
        with patch("mmf.trainers.lightning_trainer.get_mmf_env", return_value=""):
            config = self._get_config(max_steps=2, max_epochs=None)
            trainer = get_lightning_trainer(config=config)
            self.assertEqual(trainer._max_updates, 2)

            self._check_values(trainer, 0, 0)
            trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
            self._check_values(trainer, 2, 1)

    def _check_values(self, trainer, current_iteration, current_epoch):
        self.assertEqual(trainer.trainer.global_step, current_iteration)
        self.assertEqual(trainer.trainer.current_epoch, current_epoch)

    def _get_config(self, max_steps, max_epochs):
        config = {
            "trainer": {"params": {"max_steps": max_steps, "max_epochs": max_epochs}}
        }
        return get_config_with_defaults(config)
