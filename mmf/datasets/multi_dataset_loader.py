# Copyright (c) Facebook, Inc. and its affiliates.
"""
MultiDatasetLoader class is used by DatasetLoader class to load multiple datasets
and more granular
"""
import logging
import warnings

import numpy as np

from mmf.common.registry import registry
from mmf.utils.build import build_dataloader_and_sampler, build_dataset
from mmf.utils.distributed import broadcast_scalar, is_dist_initialized, is_master
from mmf.utils.general import get_batch_size

logger = logging.getLogger(__name__)


class MultiDatasetLoader:
    """
    MultiDatasetLoader class that is used for training on multiple datasets together.
    """

    def __init__(self, dataset_type="train"):
        self._dataset_type = dataset_type
        self._is_master = is_master()

        self._datasets = []
        self._loaders = []
        self._samplers = []
        self._iterators = []

        self._total_length = 0
        self._per_dataset_lengths = []
        self._num_datasets = 0
        self._finished_iterators = {}
        self._used_once = {}

    @property
    def dataset_type(self):
        return self._dataset_type

    @property
    def current_dataset_name(self):
        return self.current_dataset.name

    @property
    def num_datasets(self):
        return self._num_datasets

    @property
    def datasets(self):
        return self._datasets

    @property
    def loaders(self):
        return self._loaders

    @property
    def samplers(self):
        return self._samplers

    @property
    def iterators(self):
        return self._iterators

    @iterators.setter
    def iterators(self, iterators):
        self._iterators = iterators

    @property
    def current_dataset(self):
        return self._chosen_dataset

    # Setter only for functions which users should also be able to set
    @current_dataset.setter
    def current_dataset(self, dataset):
        self._chosen_dataset = dataset

    @property
    def current_loader(self):
        return self._chosen_loader

    @current_loader.setter
    def current_loader(self, loader):
        self._chosen_loader = loader

    @property
    def current_index(self):
        return self._loader_index

    @current_index.setter
    def current_index(self, index: int):
        self._loader_index = index

    def get_datasets(self):
        return self.datasets

    @property
    def first_loader(self):
        return self.loaders[0]

    def _process_datasets(self):
        if "datasets" not in self.config:
            logger.warning("No datasets attribute present. Setting default to vqa2.")
            datasets = "vqa2"
        else:
            datasets = self.config.datasets

        if type(datasets) == str:
            datasets = list(map(lambda x: x.strip(), datasets.split(",")))

        self._given_datasets = datasets

    def load(self, config):
        self.build_datasets(config)
        self.build_dataloaders()

    def build_datasets(self, config):
        self.config = config
        self._process_datasets()

        for dataset in self._given_datasets:
            if dataset in self.config.dataset_config:
                dataset_config = self.config.dataset_config[dataset]
            else:
                raise RuntimeError(
                    f"Dataset {dataset} is missing from " "dataset_config in config."
                )

            dataset_instance = build_dataset(dataset, dataset_config, self.dataset_type)
            if dataset_instance is None:
                continue
            self.datasets.append(dataset_instance)

            if hasattr(dataset_instance, "__len__"):
                self._per_dataset_lengths.append(len(dataset_instance))
                self._total_length += len(dataset_instance)

        self._num_datasets = len(self.datasets)
        self.current_index = 0
        self.current_dataset = self.datasets[self.current_index]

        self._infer_dataset_probabilities()

    def build_dataloaders(self):
        assert len(self._datasets) > 0, "Call build_datasets first"

        for dataset_instance in self.datasets:
            loader_instance, sampler_instance = build_dataloader_and_sampler(
                dataset_instance, self.config.training
            )

            self.loaders.append(loader_instance)
            self.samplers.append(sampler_instance)

        self.current_loader = self.loaders[self.current_index]

    def _infer_dataset_probabilities(self):
        self._dataset_probabilities = [
            1 / self._num_datasets for _ in range(self.num_datasets)
        ]

        training = self.config.get("training", {})
        self._proportional_sampling = training.get(
            "dataset_size_proportional_sampling", True
        )

        if self._dataset_type != "train":
            # If it is val or test, it needs to be all datasets need to be
            # fully iterated as metrics will be calculated in eval mode
            # over complete datasets
            self._proportional_sampling = True

        if self._proportional_sampling is True and len(self._per_dataset_lengths) > 0:
            self._dataset_probabilities = self._per_dataset_lengths[:]
            self._dataset_probabilities = [
                prob / self._total_length for prob in self._dataset_probabilities
            ]

    def __len__(self):
        # Since, this is iterator, we need to return total length == number of batches
        return self._total_length // get_batch_size()

    def __iter__(self):
        if self._num_datasets == 1:
            return iter(self.loaders[0])

        # Clear off old iterators
        self.iterators = []
        for loader in self.loaders:
            self.iterators.append(iter(loader))

        self._chosen_iterator = self.iterators[self.current_index]

        return self

    def __next__(self):
        try:
            next_batch = next(self._chosen_iterator)
        except StopIteration:
            if (
                self._proportional_sampling is True
                or len(self._used_once) != self.num_datasets
            ):
                self._finished_iterators[self.current_index] = 1

                if len(self._finished_iterators) == self.num_datasets:
                    raise
                else:
                    self.change_dataloader()
                next_batch = next(self._chosen_iterator)
            else:
                raise

        self._used_once[self.current_index] = 1
        return next_batch

    def change_dataloader(self):
        if self.num_datasets <= 1:
            return
        choice = 0

        if self._is_master:
            choice = np.random.choice(
                self.num_datasets, 1, p=self._dataset_probabilities
            )[0]

            while choice in self._finished_iterators:
                choice = np.random.choice(
                    self.num_datasets, 1, p=self._dataset_probabilities
                )[0]

        choice = broadcast_scalar(choice, 0, device=registry.get("current_device"))
        self.current_index = choice
        self.current_dataset = self.datasets[self.current_index]
        self.current_loader = self.loaders[self.current_index]
        self._chosen_iterator = self.iterators[self.current_index]

    def verbose_dump(self, *args, **kwargs):
        self._chosen_dataset.verbose_dump(*args, **kwargs)

    def prepare_batch(self, batch):
        if not hasattr(self._chosen_dataset, "prepare_batch"):
            warnings.warn(
                f"{self._chosen_dataset.dataset_name} doesn't define 'prepare_batch' "
                + "method. You are expected to prepare and move your batch to "
                + "CUDA device yourself."
            )
        else:
            batch = self._chosen_dataset.prepare_batch(batch)

        self.change_dataloader()
        return batch

    def seed_sampler(self, epoch):
        if is_dist_initialized():
            for sampler in self._samplers:
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
