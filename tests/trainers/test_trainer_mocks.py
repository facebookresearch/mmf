# Copyright (c) Facebook, Inc. and its affiliates.

import functools
from unittest.mock import MagicMock

import torch
from mmf.common.registry import registry
from mmf.common.sample import SampleList
from mmf.datasets.lightning_multi_datamodule import LightningMultiDataModule
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from mmf.datasets.multi_datamodule import MultiDataModule
from mmf.trainers.mmf_trainer import MMFTrainer
from omegaconf import OmegaConf
from tests.test_utils import NumbersDataset


class MultiDataModuleNumbersTestObject(MultiDataModule):
    def __init__(self, config, num_data):
        super().__init__(config)
        batch_size = config.training.batch_size
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
                "dataset_config": {"numbers": 0},
            }
        )
        self._num_data = num_data
        self.batch_size = batch_size
        self.config = config
        self.dataset_list = []
        dataset_builder = MMFDatasetBuilder(
            "numbers",
            functools.partial(NumbersDataset, num_examples=num_data, always_one=True),
        )
        dataset_builder.train_dataloader = self._get_dataloader_train
        dataset_builder.val_dataloader = self._get_dataloader_val
        dataset_builder.test_dataloader = self._get_dataloader_test
        self.datamodules = {"numbers": dataset_builder}

    def _get_dataloader_train(self):
        return self._get_dataloader()

    def _get_dataloader_val(self):
        return self._get_dataloader("val")

    def _get_dataloader_test(self):
        return self._get_dataloader("test")

    def _get_dataloader(self, dataset_type="train"):
        dataset = NumbersDataset(
            self._num_data, always_one=True, dataset_type=dataset_type
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )


class LightningMultiDataModuleNumbersTestObject(LightningMultiDataModule):
    def __init__(self, config, num_data):
        super().__init__(config)
        batch_size = config.training.batch_size
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
                "dataset_config": {"numbers": 0},
            }
        )
        self._num_data = num_data
        self.batch_size = batch_size
        self.config = config
        self.dataset_list = []
        dataset_builder = MMFDatasetBuilder(
            "numbers",
            functools.partial(NumbersDataset, num_examples=num_data, always_one=True),
        )
        dataset_builder.train_dataloader = self._get_dataloader_train
        dataset_builder.val_dataloader = self._get_dataloader_val
        dataset_builder.test_dataloader = self._get_dataloader_test
        self.datamodules = {"numbers": dataset_builder}

    def _get_dataloader_train(self):
        return self._get_dataloader()

    def _get_dataloader_val(self):
        return self._get_dataloader("val")

    def _get_dataloader_test(self):
        return self._get_dataloader("test")

    def _get_dataloader(self, dataset_type="train"):
        dataset = NumbersDataset(
            self._num_data, always_one=True, dataset_type=dataset_type
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )


class TrainerTrainingLoopMock(MMFTrainer):
    def __init__(self, num_train_data=100, config=None):
        self.training_config = config.training
        self.config = config
        registry.register("config", self.config)

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.distributed = False

        self.on_batch_start = MagicMock(return_value=None)
        self.on_update_start = MagicMock(return_value=None)
        self.logistics_callback = MagicMock(return_value=None)
        self.logistics_callback.log_interval = MagicMock(return_value=None)
        self.on_batch_end = MagicMock(return_value=None)
        self.on_update_end = MagicMock(return_value=None)
        self.on_validation_start = MagicMock(return_value=None)
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.early_stop_callback = MagicMock(return_value=None)
        self.on_validation_end = MagicMock(return_value=None)
        self.metrics = MagicMock(return_value={})

        self.num_data = num_train_data
        self.run_type = self.config.get("run_type", "train")

    def load_datasets(self):
        self.dataset_loader = MultiDataModuleNumbersTestObject(
            config=self.config, num_data=self.num_data
        )
        self.dataset_loader.seed_sampler = MagicMock(return_value=None)
        self.dataset_loader.prepare_batch = lambda x: SampleList(x)

        self.train_loader = self.dataset_loader.train_dataloader()
        self.val_loader = self.dataset_loader.val_dataloader()
        self.test_loader = self.dataset_loader.test_dataloader()
