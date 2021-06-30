# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock, patch

import torch
from mmf.trainers.callbacks.lr_scheduler import LRSchedulerCallback
from tests.trainers.test_utils import (
    get_config_with_defaults,
    get_lightning_trainer,
    get_mmf_trainer,
)


class TestLightningTrainerLRSchedule(unittest.TestCase):
    def test_lr_schedule(self):
        with patch("mmf.trainers.lightning_trainer.get_mmf_env", return_value=""):
            # note, be aware some of the logic also is in the SimpleLightningModel
            config = self._get_config(max_steps=8, lr_scheduler=True)
            trainer1 = get_lightning_trainer(config=config)
            trainer1.trainer.fit(trainer1.model, trainer1.data_module.train_loader)

            config = self._get_config(max_steps=8)
            trainer2 = get_lightning_trainer(config=config)
            trainer2.trainer.fit(trainer2.model, trainer2.data_module.train_loader)

            last_model_param1 = list(trainer1.model.parameters())[-1]
            last_model_param2 = list(trainer2.model.parameters())[-1]
            self.assertFalse(torch.allclose(last_model_param1, last_model_param2))

    def test_lr_schedule_compared_to_mmf_is_same(self):
        config = get_config_with_defaults(
            {"training": {"max_updates": 8, "max_epochs": None, "lr_scheduler": True}}
        )

        mmf_trainer = get_mmf_trainer(config=config)
        mmf_trainer.lr_scheduler_callback = LRSchedulerCallback(config, mmf_trainer)
        mmf_trainer.callbacks.append(mmf_trainer.lr_scheduler_callback)
        mmf_trainer.on_update_end = mmf_trainer.lr_scheduler_callback.on_update_end
        mmf_trainer.evaluation_loop = MagicMock(return_value=(None, None))
        mmf_trainer.training_loop()

        with patch("mmf.trainers.lightning_trainer.get_mmf_env", return_value=""):
            config = self._get_config(max_steps=8, lr_scheduler=True)
            trainer = get_lightning_trainer(config=config)
            trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)

            mmf_trainer.model.to(trainer.model.device)
            last_model_param1 = list(mmf_trainer.model.parameters())[-1]
            last_model_param2 = list(trainer.model.parameters())[-1]
            self.assertTrue(torch.allclose(last_model_param1, last_model_param2))

    def _get_config(self, max_steps, lr_scheduler=False):
        config = {
            "trainer": {"params": {"max_steps": max_steps}},
            "training": {"lr_scheduler": lr_scheduler},
        }
        return get_config_with_defaults(config)
