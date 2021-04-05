# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import unittest
from unittest.mock import MagicMock, patch

import torch
from mmf.common.registry import registry
from mmf.common.sample import SampleList
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from mmf.datasets.multi_datamodule import MultiDataModule
from mmf.trainers.callbacks.lr_scheduler import LRSchedulerCallback
from mmf.trainers.mmf_trainer import MMFTrainer
from mmf.utils.general import get_batch_size
from omegaconf import OmegaConf
from tests.test_utils import NumbersDataset, SimpleModel


class MultiDataModuleNumbersTestObject(MultiDataModule):
    def __init__(self, num_data, batch_size):
        self.batch_size = batch_size
        config = OmegaConf.create(
            {
                "use_features": True,
                "annotations": {
                    "train": "not_a_real_annotations_dataset",
                    "val": "not_a_real_annotations_dataset",
                },
                "features": {
                    "train": "not_a_real_features_dataset",
                    "val": "not_a_real_features_dataset",
                },
                "dataset_config": {"simple": 0},
            }
        )
        self._num_data = num_data
        self._batch_size = batch_size
        self.config = config
        self.dataset_list = []
        dataset_builder = MMFDatasetBuilder(
            "simple", functools.partial(NumbersDataset, num_examples=num_data)
        )
        dataset_builder.train_dataloader = self._get_dataloader
        dataset_builder.val_dataloader = self._get_dataloader
        dataset_builder.test_dataloader = self._get_dataloader
        self.datamodules = {"simple": dataset_builder}

    def _get_dataloader(self):
        dataset = NumbersDataset(self._num_data)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )
        return dataloader


class TrainerTrainingLoopMock(MMFTrainer):
    def __init__(
        self,
        num_train_data,
        max_updates,
        max_epochs,
        config=None,
        optimizer=None,
        update_frequency=1,
        batch_size=1,
        batch_size_per_device=None,
        fp16=False,
        on_update_end_fn=None,
        scheduler_config=None,
        grad_clipping_config=None,
        tensorboard=False,
    ):
        if config is None:
            self.config = OmegaConf.create(
                {
                    "training": {
                        "detect_anomaly": False,
                        "evaluation_interval": 10000,
                        "update_frequency": update_frequency,
                        "fp16": fp16,
                        "batch_size": batch_size,
                        "batch_size_per_device": batch_size_per_device,
                        "tensorboard": tensorboard,
                    }
                }
            )
            self.training_config = self.config.training
        else:
            config.training.batch_size = batch_size
            config.training.fp16 = fp16
            config.training.update_frequency = update_frequency
            config.training.tensorboard = tensorboard
            self.training_config = config.training
            self.config = config

        registry.register("config", self.config)

        if max_updates is not None:
            self.training_config["max_updates"] = max_updates
        if max_epochs is not None:
            self.training_config["max_epochs"] = max_epochs
        self.model = SimpleModel({"in_dim": 1})
        self.model.build()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.distributed = False

        if optimizer is None:
            self.optimizer = MagicMock()
            self.optimizer.step = MagicMock(return_value=None)
            self.optimizer.zero_grad = MagicMock(return_value=None)
        else:
            self.optimizer = optimizer

        if scheduler_config:
            config.training.lr_scheduler = True
            config.scheduler = scheduler_config
            self.lr_scheduler_callback = LRSchedulerCallback(config, self)
            self.callbacks.append(self.lr_scheduler_callback)
            on_update_end_fn = (
                on_update_end_fn
                if on_update_end_fn
                else self.lr_scheduler_callback.on_update_end
            )
        if grad_clipping_config:
            self.training_config.clip_gradients = True
            self.training_config.max_grad_l2_norm = grad_clipping_config[
                "max_grad_l2_norm"
            ]
            self.training_config.clip_norm_mode = grad_clipping_config["clip_norm_mode"]

        self.on_batch_start = MagicMock(return_value=None)
        self.on_update_start = MagicMock(return_value=None)
        self.logistics_callback = MagicMock(return_value=None)
        self.logistics_callback.log_interval = MagicMock(return_value=None)
        self.on_batch_end = MagicMock(return_value=None)
        self.on_update_end = (
            on_update_end_fn if on_update_end_fn else MagicMock(return_value=None)
        )
        self.after_training_loop = MagicMock(return_value=None)
        self.on_validation_start = MagicMock(return_value=None)
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.early_stop_callback = MagicMock(return_value=None)
        self.on_validation_end = MagicMock(return_value=None)
        self.metrics = MagicMock(return_value={})

        self.num_data = num_train_data

    def load_datasets(self):
        self.dataset_loader = MultiDataModuleNumbersTestObject(
            num_data=self.num_data, batch_size=self.config.training.batch_size
        )
        self.dataset_loader.seed_sampler = MagicMock(return_value=None)
        self.dataset_loader.prepare_batch = lambda x: SampleList(x)

        self.train_loader = self.dataset_loader.train_dataloader()
        self.val_loader = self.dataset_loader.val_dataloader()
        self.test_loader = self.dataset_loader.test_dataloader()


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
