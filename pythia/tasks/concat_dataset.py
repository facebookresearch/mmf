import functools
import types

from torch.utils.data import ConcatDataset


class PythiaConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self._dir_representation = dir(self)
    def __getattr__(self, name):
        if name in self._dir_representation:
            return getattr(self, name)
        elif hasattr(self.datasets[0], name):
            attr = getattr(self.datasets[0], name)
            # Check if the current attribute is class method function
            if isinstance(attr, types.MethodType):
                # if it is the, we to call this function for
                # each of the child datasets
                attr = functools.partial(self._call_all_datasets_func, name)
            return attr
        else:
            raise AttributeError(name)


    def _call_all_datasets_func(self, name, *args, **kwargs):
        for dataset in self.datasets:
            value = getattr(dataset, name)(*args, **kwargs)
            if value is not None:
                raise RuntimeError("Functions returning values can't be "
                                   "called through PythiaConcatDataset")
