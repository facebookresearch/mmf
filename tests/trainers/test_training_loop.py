# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock

import torch
from mmf.common.sample import SampleList
from mmf.trainers.core.profiling import TrainerProfilingMixin
from mmf.trainers.core.training_loop import TrainerTrainingLoopMixin
from omegaconf import OmegaConf
from tests.test_utils import NumbersDataset, SimpleModel


class TrainerTrainingLoopMock(TrainerTrainingLoopMixin, TrainerProfilingMixin):
    def __init__(self, num_train_data, max_updates, max_epochs):
        self.training_config = OmegaConf.create(
            {
                "detect_anomaly": False,
                "evaluation_interval": 10000,
                "update_frequency": 1,
                "fp16": True,
            }
        )
        if max_updates is not None:
            self.training_config["max_updates"] = max_updates
        if max_epochs is not None:
            self.training_config["max_epochs"] = max_epochs

        self.model = SimpleModel(1)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.dataset_loader = MagicMock()
        self.dataset_loader.seed_sampler = MagicMock(return_value=None)
        self.dataset_loader.prepare_batch = lambda x: SampleList(x)
        self.optimizer = MagicMock()
        self.optimizer.step = MagicMock(return_value=None)
        self.optimizer.zero_grad = MagicMock(return_value=None)
        dataset = NumbersDataset(num_train_data)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False
        )
        self.train_loader.current_dataset = dataset
        self.on_batch_start = MagicMock(return_value=None)
        self.on_update_start = MagicMock(return_value=None)
        self.logistics_callback = MagicMock(return_value=None)
        self.logistics_callback.log_interval = MagicMock(return_value=None)
        self.on_batch_end = MagicMock(return_value=None)
        self.on_update_end = MagicMock(return_value=None)
        self.meter = MagicMock(return_value=None)
        loss_mock = MagicMock()
        loss_mock.backward = MagicMock()
        self.after_training_loop = MagicMock(return_value=None)
        self.scaler = MagicMock()
        self.scaler.scale = MagicMock(return_value=loss_mock)
        self.scaler.step = MagicMock(return_value=None)


class TestTrainingLoop(unittest.TestCase):
    def test_epoch_over_updates(self):
        trainer = TrainerTrainingLoopMock(100, 2, 0.04)
        max_updates = trainer._calculate_max_updates()
        self.assertEqual(max_updates, 4)

        self.check_values(trainer, 0, 0, 0)
        trainer.training_loop()
        self.check_values(trainer, 4, 1, 4)

    def test_fractional_epoch(self):
        trainer = TrainerTrainingLoopMock(100, None, 0.04)
        max_updates = trainer._calculate_max_updates()
        self.assertEqual(max_updates, 4)

        self.check_values(trainer, 0, 0, 0)
        trainer.training_loop()
        self.check_values(trainer, 4, 1, 4)

    def test_updates(self):
        trainer = TrainerTrainingLoopMock(100, 2, None)
        max_updates = trainer._calculate_max_updates()
        self.assertEqual(max_updates, 2)

        self.check_values(trainer, 0, 0, 0)
        trainer.training_loop()
        self.check_values(trainer, 2, 1, 2)

    def check_values(self, trainer, current_iteration, current_epoch, num_updates):
        self.assertEqual(trainer.current_iteration, current_iteration)
        self.assertEqual(trainer.current_epoch, current_epoch)
        self.assertEqual(trainer.num_updates, num_updates)
