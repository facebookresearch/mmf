# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock, patch

import torch
from tests.datasets.test_multi_datamodule import MultiDataModuleTestObject
from tests.trainers.test_training_loop import TrainerTrainingLoopMock


class MMFTrainerMock(TrainerTrainingLoopMock):
    def __init__(self, num_train_data, max_updates, max_epochs, device="cuda"):
        super().__init__(num_train_data, max_updates, max_epochs)

    def load_datasets(self):
        self.dataset_loader = MultiDataModuleTestObject(batch_size=1)
        self.train_loader = self.dataset_loader.train_dataloader()
        self.val_loader = self.dataset_loader.val_dataloader()
        self.test_loader = self.dataset_loader.test_dataloader()


class TestEvalLoop(unittest.TestCase):
    def test_eval_loop(self):
        torch.random.manual_seed(2)
        with patch(
            "mmf.common.test_reporter.PathManager",
            return_value=MagicMock(return_value=None),
        ):
            trainer = MMFTrainerMock(100, 2, 2)
            trainer.load_datasets()
            combined_report, meter = trainer.evaluation_loop("val")
            self.assertAlmostEqual(combined_report["losses"]["loss"], 493377.5312)
            self.assertAlmostEqual(combined_report["logits"].item(), -0.2379742, 6)
