# Copyright (c) Facebook, Inc. and its affiliates.

from torch.utils.data.dataset import Subset


class MMFSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self._dir_representation = dir(self)

    def __getattr__(self, name):
        if "_dir_representation" in self.__dict__ and name in self._dir_representation:
            return getattr(self, name)
        elif "dataset" in self.__dict__ and hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        else:
            raise AttributeError(name)
