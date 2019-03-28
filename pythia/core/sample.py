import torch

from collections import OrderedDict
from copy import deepcopy


class Sample(OrderedDict):
    def __init__(self, init_dict={}):
        super().__init__(init_dict)

    def __setitem__(self, key, value):
        if key in self:
            raise AttributeError("Trying to set already set key {} to value "
                                 "{}".format(key, value))
        else:
            self[key] = value

    def __getitem__(self, key):
        if key not in self:
            raise AttributeError("Key {} not found in the sample".format(key))
        else:
            return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        return self[key]

    def fields(self):
        return list(self.keys())


class SampleList(OrderedDict):
    def __init__(self, samples=[]):
        super().__init__(self)

        if len(samples) == 0:
            return

        fields = samples[0].keys()

        for field in fields:
            if isinstance(samples[0][field], torch.Tensor):
                size = (len(samples), *samples[0][field].size())
                self[field] = samples[0][field].new_empty(size)
            else:
                self[field] = [None for _ in range(len(samples))]

            for idx, sample in enumerate(samples):
                if len(sample[field]) != len(samples[0][field]):
                    raise RuntimeError("Fields for all samples must be"
                                       " equally sized")
                self[field][idx] = self._get_data_copy(sample[field])

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError("Key {} not found in the list."
                                 "Valid choices are {}"
                                 .format(key, self.fields()))
        else:
            return self[key]

    def __getitem__(self, item):
        fields = self.keys()

        if item in fields:
            return self[item]

        sample = Sample()

        for field in fields:
            sample[field] = self[field][item]

        return sample

    def get_item_list(self, item):
        sample = self[item]

        return SampleList([sample])

    def copy(self):
        sample_list = SampleList()

        fields = self.fields()

        for field in fields:
            sample_list.add_field(field, self[field])

        return sample_list

    def fields(self):
        return list(self.keys())

    def _get_data_copy(self, data):
        if isinstance(data, torch.Tensor):
            copy_ = data.new_tensor(data)
        else:
            copy_ = deepcopy(data)
        return copy_

    def add_field(self, field, data):
        fields = self.fields()

        if len(fields) == 0:
            self[field] = self._get_data_copy(data)
        else:
            if len(data) != len(self[fields[0]]):
                raise RuntimeError("Field to be added must have same size "
                                   "as existing fields in SampleList."
                                   "Passed size: {}, Required size: {}"
                                   .format(len(data), len(self[fields[0]])))
            self[field] = self._get_data_copy(data)

    def to(self, device):
        fields = self.keys()
        sample_list = self.copy()
        if not isinstance(device, torch.device):
            if not isinstance(device, str):
                raise TypeError("device must be either 'str' or"
                                "'torch.device' type, {} found"
                                .format(type(device)))
            device = torch.device(device)

        for field in fields:
            if hasattr(sample_list[field], "to") \
               and sample_list[field].device != device:
                sample_list[field] = sample_list[field].to(device)

        return sample_list
