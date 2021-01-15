# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional

import pytorch_lightning as pl
from mmf.datasets.multi_dataset_loader import MultiDatasetLoader
from mmf.utils.general import get_batch_size


class LightningDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = get_batch_size()

        self.train_loader = MultiDatasetLoader("train")
        self.val_loader = MultiDatasetLoader("val")
        self.test_loader = MultiDatasetLoader("test")

        self.train_loader.load(self.config)
        self.val_loader.load(self.config)
        self.test_loader.load(self.config)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
