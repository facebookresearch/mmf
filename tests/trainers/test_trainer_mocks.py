# Copyright (c) Facebook, Inc. and its affiliates.

import functools
from unittest.mock import MagicMock

import torch
from mmf.common.registry import registry
from mmf.common.sample import SampleList
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from mmf.datasets.multi_datamodule import MultiDataModule
from mmf.trainers.callbacks.lr_scheduler import LRSchedulerCallback
from mmf.trainers.mmf_trainer import MMFTrainer
from mmf.utils.configuration import load_yaml
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
            self.config = load_yaml("configs/defaults.yaml")
            self.config = OmegaConf.merge(
                self.config,
                {
                    "training": {
                        "detect_anomaly": False,
                        "evaluation_interval": 10000,
                        "update_frequency": update_frequency,
                        "fp16": fp16,
                        "batch_size": batch_size,
                        "batch_size_per_device": batch_size_per_device,
                        "tensorboard": tensorboard,
                        "run_type": "train",
                        "num_workers": 0,
                    },
                    "datasets": "",
                    "model": "",
                },
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
        self.on_validation_start = MagicMock(return_value=None)
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.early_stop_callback = MagicMock(return_value=None)
        self.on_validation_end = MagicMock(return_value=None)
        self.metrics = MagicMock(return_value={})

        self.num_data = num_train_data
        self.run_type = self.config.get("run_type", "train")

    def load_datasets(self):
        self.dataset_loader = MultiDataModuleNumbersTestObject(
            num_data=self.num_data, batch_size=self.config.training.batch_size
        )
        self.dataset_loader.seed_sampler = MagicMock(return_value=None)
        self.dataset_loader.prepare_batch = lambda x: SampleList(x)

        self.train_loader = self.dataset_loader.train_dataloader()
        self.val_loader = self.dataset_loader.val_dataloader()
        self.test_loader = self.dataset_loader.test_dataloader()
