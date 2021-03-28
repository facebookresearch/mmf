# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

from tests.trainers.lightning.test_utils import get_lightning_trainer


class TestLightningTrainer(unittest.TestCase):
    def test_epoch_over_updates(self):
        trainer = get_lightning_trainer(max_steps=2, max_epochs=0.04)
        self.assertEqual(trainer._max_updates, 4)

        self._check_values(trainer, 0, 0)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
        self._check_values(trainer, 4, 0)

    def test_fractional_epoch(self):
        trainer = get_lightning_trainer(max_steps=None, max_epochs=0.04)
        self.assertEqual(trainer._max_updates, 4)

        self._check_values(trainer, 0, 0)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
        self._check_values(trainer, 4, 0)

    def test_updates(self):
        trainer = get_lightning_trainer(max_steps=2, max_epochs=None)
        self.assertEqual(trainer._max_updates, 2)

        self._check_values(trainer, 0, 0)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
        self._check_values(trainer, 2, 0)

    def _check_values(self, trainer, current_iteration, current_epoch):
        self.assertEqual(trainer.trainer.global_step, current_iteration)
        self.assertEqual(trainer.trainer.current_epoch, current_epoch)
