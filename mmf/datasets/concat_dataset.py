# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import types

from torch.utils.data import ConcatDataset


class MMFConcatDataset(ConcatDataset):
    # These functions should only be called once even if they return nothing
    _SINGLE_CALL_FUNCS = []

    def __init__(self, datasets):
        super().__init__(datasets)
        self._dir_representation = dir(self)

    def __getattr__(self, name):
        if "_dir_representation" in self.__dict__ and name in self._dir_representation:
            return getattr(self, name)
        elif "datasets" in self.__dict__ and hasattr(self.datasets[0], name):
            attr = getattr(self.datasets[0], name)
            # Check if the current attribute is class method function
            if isinstance(attr, types.MethodType):
                # if it is the, we to call this function for
                # each of the child datasets
                attr = functools.partial(self._call_all_datasets_func, name)
            return attr
        else:
            raise AttributeError(name)

    def _get_single_call_funcs(self):
        return MMFConcatDataset._SINGLE_CALL_FUNCS

    def _call_all_datasets_func(self, name, *args, **kwargs):
        for dataset in self.datasets:
            value = getattr(dataset, name)(*args, **kwargs)
            if value is not None:
                # TODO: Log a warning here
                return value
                # raise RuntimeError("Functions returning values can't be "
                #                    "called through MMFConcatDataset")
            if (
                hasattr(dataset, "get_single_call_funcs")
                and name in dataset.get_single_call_funcs()
            ):
                return
