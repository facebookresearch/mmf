# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import patch

import torch
from tests.trainers.test_utils import get_config_with_defaults, get_lightning_trainer


class TestLightningTrainerGradAccumulate(unittest.TestCase):
    def test_grad_accumulate(self):
        with patch("mmf.trainers.lightning_trainer.get_mmf_env", return_value=""):
            config = self._get_config(
                accumulate_grad_batches=2, max_steps=2, batch_size=3
            )
            trainer1 = get_lightning_trainer(config=config)
            trainer1.trainer.fit(trainer1.model, trainer1.data_module.train_loader)

            config = self._get_config(
                accumulate_grad_batches=1, max_steps=2, batch_size=6
            )
            trainer2 = get_lightning_trainer(config=config)
            trainer2.trainer.fit(trainer2.model, trainer2.data_module.train_loader)

            for param1, param2 in zip(
                trainer1.model.parameters(), trainer2.model.parameters()
            ):
                self.assertTrue(torch.allclose(param1, param2))

    def _get_config(self, accumulate_grad_batches, max_steps, batch_size):
        config = {
            "trainer": {
                "params": {
                    "accumulate_grad_batches": accumulate_grad_batches,
                    "max_steps": max_steps,
                }
            },
            "training": {"batch_size": batch_size},
        }
        return get_config_with_defaults(config)
