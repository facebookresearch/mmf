# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock

import torch
from omegaconf import OmegaConf

from mmf.common.sample import SampleList
from mmf.trainers.core.profiling import TrainerProfilingMixin
from mmf.trainers.core.training_loop import TrainerTrainingLoopMixin

DATA_ITEM_KEY = "test"


class NumbersDataset(torch.utils.data.Dataset):
    def __init__(self, num_examples):
        self.num_examples = num_examples

    def __getitem__(self, idx):
        return {DATA_ITEM_KEY: torch.tensor(idx, dtype=torch.float32)}

    def __len__(self):
        return self.num_examples


class SimpleModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = torch.nn.Linear(size, 4)

    def forward(self, prepared_batch):
        batch = prepared_batch[DATA_ITEM_KEY]
        model_output = {"losses": {"loss": torch.sum(self.linear(batch))}}
        return model_output


class TrainerTrainingLoopMock(TrainerTrainingLoopMixin, TrainerProfilingMixin):
    def __init__(self, num_train_data, max_updates, max_epochs):
        self.training_config = OmegaConf.create(
            {"detect_anomaly": False, "evaluation_interval": 10000}
        )
        if max_updates is not None:
            self.training_config["max_updates"] = max_updates
        if max_epochs is not None:
            self.training_config["max_epochs"] = max_epochs

        self.model = SimpleModel(1)
        self.dataset_loader = MagicMock()
        self.dataset_loader.seed_sampler = MagicMock(return_value=None)
        self.dataset_loader.prepare_batch = lambda x: SampleList(x)
        self.optimizer = MagicMock()
        self.optimizer.step = MagicMock(return_value=None)
        self.optimizer.zero_grad = MagicMock(return_value=None)
        dataset = NumbersDataset(num_train_data)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )
        self.on_batch_start = MagicMock(return_value=None)
        self.logistics_callback = MagicMock(return_value=None)
        self.logistics_callback.log_interval = MagicMock(return_value=None)
        self.on_batch_end = MagicMock(return_value=None)
        self.meter = MagicMock(return_value=None)
        self.after_training_loop = MagicMock(return_value=None)


class TestTrainingLoop(unittest.TestCase):
    def test_epoch_over_updates(self):
        trainer = TrainerTrainingLoopMock(100, 2, 0.04)
        max_updates = trainer._calculate_max_updates()
        self.assertEqual(max_updates, 4)

        self.assertEqual(trainer.current_iteration, 0)
        self.assertEqual(trainer.current_epoch, 0)
        self.assertEqual(trainer.num_updates, 0)

        trainer.training_loop()
        self.assertEqual(trainer.current_iteration, 4)
        self.assertEqual(trainer.current_epoch, 1)
        self.assertEqual(trainer.num_updates, 4)

    def test_fractional_epoch(self):
        trainer = TrainerTrainingLoopMock(100, None, 0.04)
        max_updates = trainer._calculate_max_updates()
        self.assertEqual(max_updates, 4)

        self.assertEqual(trainer.current_iteration, 0)
        self.assertEqual(trainer.current_epoch, 0)
        self.assertEqual(trainer.num_updates, 0)

        trainer.training_loop()
        self.assertEqual(trainer.current_iteration, 4)
        self.assertEqual(trainer.current_epoch, 1)
        self.assertEqual(trainer.num_updates, 4)

    def test_updates(self):
        trainer = TrainerTrainingLoopMock(100, 2, None)
        max_updates = trainer._calculate_max_updates()
        self.assertEqual(max_updates, 2)

        self.assertEqual(trainer.current_iteration, 0)
        self.assertEqual(trainer.current_epoch, 0)
        self.assertEqual(trainer.num_updates, 0)

        trainer.training_loop()
        self.assertEqual(trainer.current_iteration, 2)
        self.assertEqual(trainer.current_epoch, 1)
        self.assertEqual(trainer.num_updates, 2)
