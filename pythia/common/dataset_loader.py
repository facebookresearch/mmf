# Copyright (c) Facebook, Inc. and its affiliates.
import os

import yaml
from torch.utils.data import DataLoader

from pythia.common.batch_collator import BatchCollator
from pythia.common.test_reporter import TestReporter
from pythia.datasets.multi_dataset import MultiDataset
from pythia.datasets.samplers import DistributedSampler
from pythia.utils.general import get_batch_size


class DatasetLoader:
    def __init__(self, config):
        self.config = config

    def load_datasets(self):
        self.train_dataset = MultiDataset("train")
        self.val_dataset = MultiDataset("val")
        self.test_dataset = MultiDataset("test")

        self.train_dataset.load(**self.config)
        self.val_dataset.load(**self.config)
        self.test_dataset.load(**self.config)

        if self.train_dataset.num_datasets == 1:
            self.train_loader = self.train_dataset.first_loader
            self.val_loader = self.val_dataset.first_loader
            self.test_loader = self.test_dataset.first_loader
        else:
            self.train_loader = self.train_dataset
            self.val_loader = self.val_dataset
            self.test_loader = self.test_dataset

        self.mapping = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset,
        }

        self.test_reporter = None
        self.should_not_log = self.config.training_parameters.should_not_log

    @property
    def dataset_config(self):
        return self._dataset_config

    @dataset_config.setter
    def dataset_config(self, config):
        self._dataset_config = config

    def get_config(self):
        return self._dataset_config

    def get_test_reporter(self, dataset_type):
        dataset = getattr(self, "{}_dataset".format(dataset_type))
        return TestReporter(dataset)

    def update_registry_for_model(self, config):
        self.train_dataset.update_registry_for_model(config)
        self.val_dataset.update_registry_for_model(config)
        self.test_dataset.update_registry_for_model(config)

    def clean_config(self, config):
        self.train_dataset.clean_config(config)
        self.val_dataset.clean_config(config)
        self.test_dataset.clean_config(config)

    def prepare_batch(self, batch, *args, **kwargs):
        return self.mapping[batch.dataset_type].prepare_batch(batch)

    def verbose_dump(self, report, *args, **kwargs):
        if self.config.training_parameters.verbose_dump:
            dataset_type = report.dataset_type
            self.mapping[dataset_type].verbose_dump(report, *args, **kwargs)

    def seed_sampler(self, dataset_type, seed):
        dataset = getattr(self, "{}_dataset".format(dataset_type))
        dataset.seed_sampler(seed)
