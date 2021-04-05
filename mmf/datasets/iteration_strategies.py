# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import warnings
from dataclasses import dataclass
from typing import Dict

import numpy as np
from mmf.common.registry import registry
from mmf.utils.configuration import get_global_config
from mmf.utils.dataset import dataset_list_from_config
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class IterationStrategy:
    """
    Base class for defining iteration strategies that will be used
    for iterating over multiple datasets during multitasking.

    An IterationStrategy implementation should `__call__` method
    which returns index of dataset from which next batch must be
    pulled.

    Class can also define `should_exhaust_all_iterators` property
    which defines whether all iterators should be exhausted before
    reigniting next batch of iterators. For example, in size
    proportional iteration strategy, all iterators must be finished
    before starting a new round so that all of them get equal
    opportunity to present themselves according to their size.

    Args:
        config (Config): Object of type Config which should be defined
            for each iteration strategy for configurable parameters.
        dataloaders (Dict[str, DataLoader]): A dictionary containing
            mapping from dataset key to its dataloader.

    Usage::

        from dataclasses import dataclass
        from mmf.common.registry import registry
        from mmf.datasets.iterators import IterationStrategy


        @registry.register_iteration_strategy("my_iteration_strategy")
        class MyStrategy(IterationStrategy):
            @dataclass
            class Config:
                name: str = "my_strategy"
            def __init__(self, config, dataloader):
                ...
    """

    @dataclass
    class Config:
        name: str = MISSING

    def __init__(
        self, config: Config, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        config = OmegaConf.merge(OmegaConf.structured(self.Config), config)
        self.config = config
        self.dataloaders = dataloaders

    @classmethod
    def from_params(cls, dataloaders: Dict[str, DataLoader], **kwargs):
        config = OmegaConf.structured(cls.Config(**kwargs))
        return cls(config, dataloaders)

    @property
    def should_exhaust_all_iterators(self) -> bool:
        return False

    def _check_not_epoch_training(self):
        """
        Having this allows easy override of the strategy in non-MMF
        use cases
        """
        training = get_global_config("training")
        assert (
            training.get("max_epochs", None) is None
        ), f"{self.__class__.__name__} doesn't make sense with epoch based training"

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("__call__ hasn't been implemented")


@registry.register_iteration_strategy("constant")
class ConstantIterationStrategy(IterationStrategy):
    """
    Always returns a constant number. Useful for mimicing single task
    training in multitask setup for verification or defaults purposes

    index to be returned can be specified in config parameter as `idx`.
    """

    @dataclass
    class Config(IterationStrategy.Config):
        name: str = "constant"
        idx: int = 0

    def __init__(
        self, config: Config, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        self._idx = self.config.idx

    @property
    def should_exhaust_all_iterators(self) -> bool:
        return True

    def __call__(self, *args, **kwargs):
        return self._idx


@registry.register_iteration_strategy("round_robin")
class RoundRobinIterationStrategy(IterationStrategy):
    """
    Samples datasets one by one in round robin fashion.

    Start index can be specified in config as `start_idx`.

    Also defaults to size proportional sampling as roundrobin
    doesn't make sense with validation and testing splits
    as they need to finish one complete epoch.
    """

    @dataclass
    class Config(IterationStrategy.Config):
        name: str = "round_robin"
        start_idx: int = 0

    def __init__(
        self, config: Config, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        self._check_not_epoch_training()

        if "start_idx" in self.config:
            self._current_idx = self.config.start_idx

    def __call__(self, *args, **kwargs):
        nxt = self._current_idx
        self._current_idx = (self._current_idx + 1) % len(self.dataloaders)
        return nxt


@registry.register_iteration_strategy("random")
class RandomIterationStrategy(IterationStrategy):
    """
    Samples random number each time when sampled.

    Follows test/validation strategy similar to RoundRobin.
    """

    @dataclass
    class Config(IterationStrategy.Config):
        name: str = "random"

    def __init__(
        self, config: Config, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        self._check_not_epoch_training()

    def __call__(self, *args, **kwargs):
        choice = np.random.choice(len(self.dataloaders), 1)[0]
        return choice


@registry.register_iteration_strategy("size_proportional")
class SizeProportionalIterationStrategy(IterationStrategy):
    """
    Samples index based on size of each dataset. Bigger datasets
    are sampled more and this strategy requires completing
    all iterators before starting new ones. Default in MMF.
    """

    @dataclass
    class Config(IterationStrategy.Config):
        name: str = "size_proportional"

    def __init__(
        self, config: Config, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        self._per_dataset_lengths = []
        self._total_length = 0

        for loader in self.dataloaders.values():
            # Some loaders might not have dataset attribute
            # set, in this case we need to fail gracefully as we can't
            # calculate lengths.
            assert hasattr(loader, "dataset"), (
                "loaders need dataset objects to work with "
                + "'size_proportional' sampling"
            )

            dataset_instance = loader.dataset

            assert hasattr(dataset_instance, "__len__"), (
                "all datasets should have __len__ defined "
                + "to work with proportional sampling iterator"
            )
            dataset_instance_length = len(dataset_instance)
            assert (
                dataset_instance_length
            ), f"dataset: {dataset_instance.dataset_type} is empty"
            self._per_dataset_lengths.append(dataset_instance_length)
            self._total_length += dataset_instance_length

        self._dataset_probabilities = self._per_dataset_lengths[:]
        self._dataset_probabilities = [
            prob / self._total_length for prob in self._dataset_probabilities
        ]

    def __call__(self, *args, **kwargs):
        choice = np.random.choice(
            len(self.dataloaders), 1, p=self._dataset_probabilities
        )[0]
        return choice

    @property
    def should_exhaust_all_iterators(self):
        return True


@registry.register_iteration_strategy("ratios")
class RatiosIterationStrategy(IterationStrategy):
    """
    Samples based on ratios specified as `sampling_ratios` parameter
    in the config. Default to validation/test strategy as in RoundRobin.

    `sampling_ratios` defines a dictionary pointing from dataset key to
    a floating ration specifying how much the dataset should be sampled.
    Floats together should sum to one.

    `datasets` is a list of datasets that would be sampled. This should
    a subset or same as `sampling_ratios.keys()`.
    """

    @dataclass
    class Config(IterationStrategy.Config):
        name: str = "ratios"
        sampling_ratios: Dict[str, float] = MISSING

    def __init__(
        self, config: Config, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        self._check_not_epoch_training()
        given_datasets = self._get_given_datasets()
        sampling_ratios = self.config.get("sampling_ratios", {})
        probabilities = []
        for dataset in given_datasets:
            assert (
                dataset in sampling_ratios
            ), f"{dataset} must be specified in sampling_ratios param for multitasking"
            probabilities.append(sampling_ratios[dataset])

        # normalize the sampling ratios to sum up to 1
        prob_sum = sum(probabilities)
        assert all(prob >= 0 for prob in probabilities) and prob_sum > 0, (
            "sampling_ratios param for multitasking must be all non-negative "
            "and at least one of them needs to be positive."
        )

        self._dataset_probabilities = [prob / prob_sum for prob in probabilities]
        logger.info("Using per-dataset sampling probabilities:")
        for dataset, prob in zip(given_datasets, self._dataset_probabilities):
            logger.info(f"\t{dataset}: {prob}")

    def __call__(self, *args, **kwargs):
        choice = np.random.choice(
            len(self.dataloaders), 1, p=self._dataset_probabilities
        )[0]
        return choice

    def _get_given_datasets(self):
        config = registry.get("config")
        datasets = None
        if config is not None and "datasets" not in config:
            datasets = dataset_list_from_config(config)

        if datasets is None or len(datasets) == 0:
            warnings.warn(
                "Either 'datasets' key not in global config or is a empty list. "
                + "Moving forward with dataset list same as sampling ratios"
            )
            return list(self.config.get("sampling_ratios", {}).keys())
        else:
            return datasets
