# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import patch

import torch
from tests.trainers.test_utils import get_lightning_trainer


class TestLightningTrainerGradAccumulate(unittest.TestCase):
    def test_grad_accumulate(self):
        with patch("mmf.trainers.lightning_trainer.get_mmf_env", return_value=None):
            trainer1 = get_lightning_trainer(
                accumulate_grad_batches=2, max_steps=2, batch_size=3
            )
            trainer1.trainer.fit(trainer1.model, trainer1.data_module.train_loader)

            trainer2 = get_lightning_trainer(
                accumulate_grad_batches=1, max_steps=2, batch_size=6
            )
            trainer2.trainer.fit(trainer2.model, trainer2.data_module.train_loader)

            for param1, param2 in zip(
                trainer1.model.parameters(), trainer2.model.parameters()
            ):
                self.assertTrue(torch.allclose(param1, param2))
