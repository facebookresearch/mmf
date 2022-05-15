# Copyright (c) Facebook, Inc. and its affiliates.
"""
MultiDatasetLoader class is used by DatasetLoader class to load multiple datasets
and more granular
"""
import logging
import warnings
from typing import Dict, Iterator

import torch
from mmf.common.sample import convert_batch_to_sample_list, SampleList
from mmf.datasets import iteration_strategies
from mmf.utils.build import build_dataloader_and_sampler, build_dataset
from mmf.utils.dataset import dataset_list_from_config
from mmf.utils.distributed import (
    broadcast_scalar,
    get_world_size,
    is_dist_initialized,
    is_main,
    is_xla,
)
from mmf.utils.general import get_batch_size, get_current_device
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader, Sampler


logger = logging.getLogger(__name__)


class MultiDataLoader:
    def __init__(
        self,
        loaders: Dict[str, DataLoader],
        iteration_strategy: iteration_strategies.IterationStrategy = None,
    ):
        if loaders is None or len(loaders) == 0:
            warnings.warn(
                "Empty loaders passed into MultiDataLoader. This can have "
                "unintended consequences."
            )

        if iteration_strategy is None:
            iteration_strategy = iteration_strategies.RoundRobinIterationStrategy(
                OmegaConf.create(), loaders
            )

        self._iteration_strategy = iteration_strategy
        self._loaders = loaders
        self._is_main = is_main()
        self._num_datasets = len(self.loaders)
        self.dataset_list = list(loaders.keys())
        self._iterators = {}
        self._finished_iterators = {}

        self.current_index = 0
        self.set_lengths()
        self.set_samplers()

    def set_lengths(self):
        self._total_length = 0
        for loader in self.loaders.values():
            # Some loaders might not have dataset attribute
            # set, in this case we won't consider them in
            # dataset lengths.
            if not hasattr(loader, "dataset"):
                continue

            dataset_instance = loader.dataset

            if hasattr(dataset_instance, "__len__"):
                dataset_instance_length = len(dataset_instance)
                assert dataset_instance_length, f"dataset: {self.dataset_type} is empty"
                self._total_length += dataset_instance_length

    def set_samplers(self):
        self.samplers: Dict[str, Sampler] = {}
        for key, loader in self.loaders.items():
            if hasattr(loader, "sampler"):
                self.samplers[key] = loader.sampler

    def get_datasets(self):
        return [loader.dataset for loader in self.loaders.values()]

    @property
    def loaders(self) -> Dict[str, DataLoader]:
        return self._loaders

    @property
    def samplers(self) -> Dict[str, Sampler]:
        return self._samplers

    @samplers.setter
    def samplers(self, samplers: Dict[str, Sampler]):
        self._samplers = samplers

    @property
    def num_datasets(self) -> int:
        return self._num_datasets

    @property
    def iterators(self) -> Dict[str, Iterator[SampleList]]:
        return self._iterators

    @iterators.setter
    def iterators(self, iterators: Dict[str, Iterator[SampleList]]):
        self._iterators = iterators

    @property
    def current_loader(self) -> DataLoader:
        return self.loaders[self.current_dataset_name]

    @property
    def iteration_strategy(self) -> iteration_strategies.IterationStrategy:
        return self._iteration_strategy

    @property
    def current_iterator(self) -> DataLoader:
        return self.iterators[self.current_dataset_name]

    @property
    def current_dataset_name(self) -> str:
        return self.dataset_list[self.current_index]

    @property
    def current_dataset(self) -> torch.utils.data.Dataset:
        if hasattr(self.current_loader, "dataset"):
            return self.current_loader.dataset
        else:
            return None

    @property
    def first_loader(self) -> DataLoader:
        return list(self.loaders.values())[0]

    def __len__(self) -> int:
        # Since, this is iterator, we need to return total length == number of batches
        # and as get_batch_size returns per GPU batch size, it needs to be multiplied
        # by world size
        batch_size = get_batch_size() * get_world_size()
        # Changed the length to accomadate drop_last == True
        # drop_last is required if the batch is split into multiple cores
        # some of the cores may not have enough examples.
        if is_xla():
            logging.info(
                "drop_last is set to True to avoid uneven dimension shapes "
                "across cores."
            )
            return (self._total_length) // batch_size
        else:
            # This assumes drop_last=False for all loaders. See also
            # build_dataloader_and_sampler().
            return (self._total_length + batch_size - 1) // batch_size

    def __iter__(self):
        # Clear off old iterators
        self._finished_iterators = {}
        self.iterators = {}

        for key, loader in self.loaders.items():
            self.iterators[key] = iter(loader)

        self.change_dataloader()

        return self

    def __next__(self) -> SampleList:
        """Calculation of next batch is performed using following logic.

        Current chosen iterator is set in the change_dataloader function
        based on the chosen iteration strategy which is called everytime
        prepare_batch is called.

        If we get the next batch from iterator without any StopIteration exception,
        we return it as it is. Otherwise, we have two cases:

        1. In some iteration strategies (example size proportional), each dataset
        needs to same number of epochs at any given time, we need to yield
        StopIteration exception when all iterators are finished. In turn, this
        will yield to __iter__ all reignite all of the iterators. The code will
        not reach __iter__ until unless all iterators are exhausted. An iteration
        strategy should specify this behavior through `should_exhaust_all_iterators`
        property

        2. In other cases of iteration strategies, epochs don't make sense.
        Think of a case of random (equal) proportional sampling for dataset x and y
        where x is half the size of y. When x will complete its 2 epochs, y will
        have only 1 epoch completed. **So please don't use max_epochs or epoch
        based training in this case as it won't be honored**. If an iterator is
        finished, we just reignite it in this case and finished iterators
        variable isn't used. This means that this case will never reach the
        __iter__ function ever again.


        Returns:
            SampleList: sample list instance from currently selected dataset
        """
        try:
            next_batch = next(self.current_iterator)
        except StopIteration:
            if self.iteration_strategy.should_exhaust_all_iterators:
                self._finished_iterators[self.current_dataset_name] = 1

                if len(self._finished_iterators) == self.num_datasets:
                    raise
                else:
                    self.change_dataloader()
                next_batch = next(self.current_iterator)
            else:
                iterator = iter(self.current_loader)
                self.iterators[self.current_dataset_name] = iterator
                next_batch = next(self.current_iterator)

        # Save dataset name and dataset type beforehand as
        # prepare_data will change the current index
        current_dataset_name = self.current_dataset_name
        current_dataset_type = self.current_dataset.dataset_type

        next_batch = self.prepare_batch(next_batch)
        next_batch = convert_batch_to_sample_list(next_batch)

        next_batch.dataset_name = current_dataset_name
        next_batch.dataset_type = current_dataset_type
        return next_batch

    def change_dataloader(self):
        choice = 0

        if self.num_datasets <= 1:
            self.current_index = choice
            return

        if self._is_main:
            choice = self.iteration_strategy()

            # self._finished_iterators will always be empty in case of
            # non-proportional (equal) sampling
            while self.dataset_list[choice] in self._finished_iterators:
                choice = self.iteration_strategy()

        choice = broadcast_scalar(choice, 0, device=get_current_device())
        self.current_index = choice

    def prepare_batch(self, batch: SampleList) -> SampleList:
        if self.current_dataset and hasattr(self.current_dataset, "prepare_batch"):
            batch = self.current_dataset.prepare_batch(batch)

        self.change_dataloader()
        return batch

    def seed_sampler(self, epoch: int):
        if is_dist_initialized():
            for sampler in self.samplers.values():
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)


# TODO: Deprecate in favor of MultiDataModule
class MultiDatasetLoader(MultiDataLoader):
    """
    MultiDatasetLoader class that is used for training on multiple datasets together.
    """

    def __init__(self, dataset_type: str = "train"):
        self._dataset_type = dataset_type
        self._datasets = []
        super().__init__({})

    @property
    def dataset_type(self):
        return self._dataset_type

    @property
    def datasets(self):
        return self._datasets

    def load(self, config):
        self.build_datasets(config)
        self.build_dataloaders()
        self.set_lengths()

    def build_datasets(self, config):
        self._datasets = []
        self.config = config
        self._given_datasets = dataset_list_from_config(self.config)

        for dataset in self._given_datasets:
            if dataset in self.config.dataset_config:
                dataset_config = self.config.dataset_config[dataset]
            else:
                warnings.warn(
                    f"Dataset {dataset} is missing from dataset_config"
                    + " in config. Proceeding with empty config."
                )
                dataset_config = OmegaConf.create()

            dataset_instance = build_dataset(dataset, dataset_config, self.dataset_type)
            if dataset_instance is None:
                continue
            self.datasets.append(dataset_instance)
            self.dataset_list.append(dataset)

        self._num_datasets = len(self.datasets)
        self.current_index = 0

        self._infer_dataset_probabilities()

    def build_dataloaders(self):
        assert len(self._datasets) > 0, "Call build_datasets first"

        for dataset_instance in self.datasets:
            loader_instance, _ = build_dataloader_and_sampler(
                dataset_instance, self.config.training
            )
            sampler_instance = loader_instance.sampler
            self.loaders[dataset_instance.name] = loader_instance
            self.samplers[dataset_instance.name] = sampler_instance

        self.current_loader = self.loaders[self.current_dataset_name]

    def verbose_dump(self, *args, **kwargs):
        self._chosen_dataset.verbose_dump(*args, **kwargs)

    # Kept for backwards compatibility for now
    # TODO: Remove in future.
    def _infer_dataset_probabilities(self):
        from mmf.utils.configuration import get_global_config

        training = get_global_config("training")

        proportional_sampling = training.get("dataset_size_proportional_sampling", True)

        if proportional_sampling is True:
            strategy = iteration_strategies.SizeProportionalIterationStrategy
            self._iteration_strategy = strategy(OmegaConf.create(), self.loaders)
        else:
            self._iteration_strategy = iteration_strategies.RandomIterationStrategy(
                OmegaConf.create(), self.loaders
            )

        multitasking = get_global_config("multitasking")
        multitasking_enabled = multitasking.get("enabled", False)

        assert (
            proportional_sampling is True or training.get("max_epochs", None) is None
        ), "Epoch based training can only be used with size proportional sampling"

        assert not (proportional_sampling and multitasking_enabled), (
            "Multitasking (manually-specified) per-dataset ratios cannot be used "
            "with size proportional sampling"
        )

        if multitasking_enabled and "sampling_ratios" in multitasking:
            self._iteration_strategy = iteration_strategies.RatiosIterationStrategy(
                OmegaConf.create(
                    {
                        "sampling_ratios": multitasking.sampling_ratios,
                        "datasets": self._given_datasets,
                    }
                ),
                self._loaders,
            )
        elif proportional_sampling is True:
            strategy = iteration_strategies.SizeProportionalIterationStrategy
            self._iteration_strategy = strategy(OmegaConf.create(), self.loaders)
        else:
            self._iteration_strategy = iteration_strategies.RandomIterationStrategy(
                OmegaConf.create(), self.loaders
            )
