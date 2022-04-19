# Copyright (c) Facebook, Inc. and its affiliates.
import functools

import torch
from mmf.datasets.lightning_multi_datamodule import LightningMultiDataModule
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from mmf.datasets.multi_datamodule import MultiDataModule
from omegaconf import OmegaConf
from tests.datasets.test_mmf_dataset_builder import SimpleMMFDataset


class MultiDataModuleTestObject(MultiDataModule):
    def __init__(self, batch_size):
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
        self.config = config
        self.dataset_list = []
        dataset_builder = MMFDatasetBuilder(
            "simple", functools.partial(SimpleMMFDataset, num_examples=100)
        )
        dataset_builder.train_dataloader = self._get_dataloader
        dataset_builder.val_dataloader = self._get_dataloader
        dataset_builder.test_dataloader = self._get_dataloader
        self.datamodules = {"simple": dataset_builder}

    def _get_dataloader(self):
        dataset = SimpleMMFDataset(
            num_examples=100,
            dataset_name="simple",
            dataset_type="val",
            config=self.config,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )
        return dataloader


class LightningDataModuleTestObject(LightningMultiDataModule):
    def __init__(self, batch_size):
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
        self.config = config
        self.dataset_list = []
        dataset_builder = MMFDatasetBuilder(
            "simple", functools.partial(SimpleMMFDataset, num_examples=100)
        )
        dataset_builder.train_dataloader = self._get_dataloader
        dataset_builder.val_dataloader = self._get_dataloader
        dataset_builder.test_dataloader = self._get_dataloader
        self.datamodules = {"simple": dataset_builder}

    def _get_dataloader(self):
        dataset = SimpleMMFDataset(
            num_examples=100,
            dataset_name="simple",
            dataset_type="val",
            config=self.config,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )
        return dataloader
