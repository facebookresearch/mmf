# Copyright (c) Facebook, Inc. and its affiliates.
import math
from typing import Dict

from mmf.datasets import iteration_strategies
from mmf.datasets.multi_dataset_loader import MultiDataLoader
from torch.utils.data import DataLoader


class LightningMultiDataLoader(MultiDataLoader):
    """
    LightningMultiDataLoader class is used by DatasetLoader class to load multiple
    datasets and more granular. This class overrides some functions from MultiDataLoader
    to make them lightning trainer compatible
    """

    def __init__(
        self,
        loaders: Dict[str, DataLoader],
        iteration_strategy: iteration_strategies.IterationStrategy = None,
    ):
        super().__init__(loaders, iteration_strategy)

    def has_len(self):
        for loader in self.loaders.values():
            if not hasattr(loader, "dataset"):
                continue
            dataset_instance = loader.dataset
            if not hasattr(dataset_instance, "__len__"):
                return False
        return True

    def set_lengths(self):
        self._total_length = 0

        if not self.has_len():
            self._total_length = math.inf
            return

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
