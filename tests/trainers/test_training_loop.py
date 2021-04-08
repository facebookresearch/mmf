# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import patch

import torch
from mmf.common.registry import registry
from mmf.utils.general import get_batch_size
from tests.test_utils import SimpleModel
from tests.trainers.test_trainer_mocks import TrainerTrainingLoopMock


class TestTrainingLoop(unittest.TestCase):
    def test_update_frequency_num_remaining_updates_greater_than_update_frequency(self):
        trainer1 = self._train_with_condition(
            num_train_data=20,
            max_updates=None,
            max_epochs=2,
            update_frequency=3,
            batch_size=6,
        )
        self.assertEqual(trainer1.num_updates, 4)

        trainer2 = self._train_with_condition(
            num_train_data=20,
            max_updates=4,
            max_epochs=None,
            update_frequency=1,
            batch_size=18,
        )
        self.assertEqual(trainer2.num_updates, 4)
        self._compare_model_params(trainer1, trainer2)

    def test_update_frequency_reporting(self):
        def _on_update_end(report, meter, should_log):
            # the losses here should be the sum of two losses in
            # iteration 0 and iteration 1 (both constitute update 0).
            # Here iter 1 loss: 0.2599, iter 2 loss: 4.2090
            loss = report.losses["loss"].detach().cpu().item()
            self.assertAlmostEqual(loss, 4.4688, 4)

        self._train_with_condition(
            num_train_data=100,
            max_updates=1,
            max_epochs=None,
            update_frequency=2,
            batch_size=2,
            on_update_end_fn=_on_update_end,
        )

    def test_update_frequency_correct_final_iteration(self):
        trainer = TrainerTrainingLoopMock(100, 2, None, update_frequency=2)
        trainer.load_datasets()
        trainer.training_loop()
        self.assertEqual(trainer.max_updates, 2)
        self.assertEqual(trainer.current_iteration, 4)

    def test_update_frequency_same_model_params(self):
        trainer1 = self._train_with_condition(
            num_train_data=100,
            max_updates=2,
            max_epochs=None,
            update_frequency=2,
            batch_size=2,
        )
        trainer1.load_datasets()
        trainer2 = self._train_with_condition(
            num_train_data=100,
            max_updates=2,
            max_epochs=None,
            update_frequency=1,
            batch_size=4,
        )
        trainer2.load_datasets()
        self._compare_model_params(trainer1, trainer2)

    def _compare_model_params(self, trainer1, trainer2):
        for param1, param2 in zip(
            trainer1.model.parameters(), trainer2.model.parameters()
        ):
            self.assertTrue(torch.allclose(param1, param2))

    def _train_with_condition(
        self,
        num_train_data,
        max_updates,
        max_epochs,
        update_frequency,
        batch_size,
        on_update_end_fn=None,
    ):
        torch.random.manual_seed(2)
        model = SimpleModel({"in_dim": 1})
        model.build()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = TrainerTrainingLoopMock(
            num_train_data,
            max_updates,
            max_epochs,
            optimizer=opt,
            update_frequency=update_frequency,
            batch_size=batch_size,
        )
        trainer.load_datasets()
        if on_update_end_fn:
            trainer.on_update_end = on_update_end_fn
        model.to(trainer.device)
        trainer.model = model
        trainer.training_loop()
        return trainer

    def test_epoch_over_updates(self):
        trainer = TrainerTrainingLoopMock(100, 2, 0.04)
        trainer.load_datasets()
        max_updates = trainer._calculate_max_updates()
        self.assertEqual(max_updates, 4)

        self.check_values(trainer, 0, 0, 0)
        trainer.training_loop()
        self.check_values(trainer, 4, 1, 4)

    def test_fractional_epoch(self):
        trainer = TrainerTrainingLoopMock(100, None, 0.04)
        trainer.load_datasets()
        max_updates = trainer._calculate_max_updates()
        self.assertEqual(max_updates, 4)

        self.check_values(trainer, 0, 0, 0)
        trainer.training_loop()
        self.check_values(trainer, 4, 1, 4)

    def test_updates(self):
        trainer = TrainerTrainingLoopMock(100, 2, None)
        trainer.load_datasets()
        max_updates = trainer._calculate_max_updates()
        self.assertEqual(max_updates, 2)

        self.check_values(trainer, 0, 0, 0)
        trainer.training_loop()
        self.check_values(trainer, 2, 1, 2)

    def test_batch_size_per_device(self):
        # Need to patch the mmf.utils.general's world size not mmf.utils.distributed
        # as the first one is what will be used
        with patch("mmf.utils.general.get_world_size", return_value=2):
            trainer = TrainerTrainingLoopMock(100, 2, None, batch_size=4)
            registry.register("config", trainer.config)
            batch_size = get_batch_size()
            trainer.config.training.batch_size = batch_size
            trainer.load_datasets()
            # Train loader has batch size per device, for global batch size 4
            # with world size 2, batch size per device should 4 // 2 = 2
            self.assertEqual(trainer.train_loader.current_loader.batch_size, 2)
            # This is per device, so should stay same
            trainer = TrainerTrainingLoopMock(100, 2, None, batch_size_per_device=4)
            registry.register("config", trainer.config)
            batch_size = get_batch_size()
            trainer.config.training.batch_size = batch_size
            trainer.load_datasets()
            self.assertEqual(trainer.train_loader.current_loader.batch_size, 4)

        max_updates = trainer._calculate_max_updates()
        self.assertEqual(max_updates, 2)

        self.check_values(trainer, 0, 0, 0)
        trainer.training_loop()
        self.check_values(trainer, 2, 1, 2)

    def check_values(self, trainer, current_iteration, current_epoch, num_updates):
        self.assertEqual(trainer.current_iteration, current_iteration)
        self.assertEqual(trainer.current_epoch, current_epoch)
        self.assertEqual(trainer.num_updates, num_updates)
