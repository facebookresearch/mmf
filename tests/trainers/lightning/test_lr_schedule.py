# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock

import torch
from tests.trainers.lightning.test_utils import (
    get_lightning_trainer,
    get_mmf_trainer,
    get_trainer_config,
)


class TestLightningTrainerLRSchedule(unittest.TestCase):
    def test_lr_schedule(self):
        # note, be aware some of the logic also is in the SimpleLightningModel
        trainer1 = get_lightning_trainer(max_steps=8, lr_scheduler=True)
        trainer1.trainer.fit(trainer1.model, trainer1.data_module.train_loader)

        trainer2 = get_lightning_trainer(max_steps=8)
        trainer2.trainer.fit(trainer2.model, trainer2.data_module.train_loader)

        last_model_param1 = list(trainer1.model.parameters())[-1]
        last_model_param2 = list(trainer2.model.parameters())[-1]
        self.assertFalse(torch.allclose(last_model_param1, last_model_param2))

    def test_lr_schedule_compared_to_mmf_is_same(self):
        trainer_config = get_trainer_config()
        mmf_trainer = get_mmf_trainer(
            max_updates=8, max_epochs=None, scheduler_config=trainer_config.scheduler
        )
        mmf_trainer.evaluation_loop = MagicMock(return_value=(None, None))
        mmf_trainer.training_loop()

        trainer = get_lightning_trainer(max_steps=8, lr_scheduler=True)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)

        mmf_trainer.model.to(trainer.model.device)
        last_model_param1 = list(mmf_trainer.model.parameters())[-1]
        last_model_param2 = list(trainer.model.parameters())[-1]
        self.assertTrue(torch.allclose(last_model_param1, last_model_param2))
