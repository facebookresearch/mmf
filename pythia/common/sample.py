import torch
import collections

from collections import OrderedDict
from copy import deepcopy


class Sample(OrderedDict):
    def __init__(self, init_dict={}):
        super().__init__(init_dict)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        return self[key]

    def fields(self):
        return list(self.keys())


class SampleList(OrderedDict):
    _TENSOR_FIELD_ = "_tensor_field"

    def __init__(self, samples=[]):
        super().__init__(self)

        if len(samples) == 0:
            return

        if self._check_and_load_dict(samples):
            return
        # If passed sample list was in form of key, value pairs of tuples
        # return after loading these
        if self._check_and_load_tuple(samples):
            return

        fields = samples[0].keys()

        for field in fields:
            if isinstance(samples[0][field], torch.Tensor):
                size = (len(samples), *samples[0][field].size())
                self[field] = samples[0][field].new_empty(size)
                if self._get_tensor_field() is None:
                    self._set_tensor_field(field)
            else:
                self[field] = [None for _ in range(len(samples))]

            for idx, sample in enumerate(samples):
                # it should be a tensor but not a 0-d tensor
                if isinstance(sample[field], torch.Tensor) and \
                    len(sample[field].size()) != 0 and \
                    sample[field].size(0) != samples[0][field].size(0):
                    raise AssertionError("Fields for all samples must be"
                                         " equally sized.")

                self[field][idx] = self._get_data_copy(sample[field])

            if isinstance(samples[0][field], collections.Mapping):
                self[field] = SampleList(self[field])

    def _check_and_load_tuple(self, samples):
        if isinstance(samples[0], (tuple, list)) \
            and isinstance(samples[0][0], str):
            for kv_pair in samples:
                self.add_field(kv_pair[0], kv_pair[1])
            return True
        else:
            return False

    def _check_and_load_dict(self, samples):
        if isinstance(samples, collections.Mapping):
            for key, value in samples.items():
                self.add_field(key, value)
            return True
        else:
            return False

    def _fix_sample_type(self, samples):
        if not isinstance(samples[0], Sample):
            proper_samples = []
            for sample in samples:
                proper_samples.append(Sample(sample))
            samples = proper_samples
        return samples

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError("Key {} not found in the SampleList. "
                                 "Valid choices are {}"
                                 .format(key, self.fields()))
        fields = self.keys()

        if key in fields:
            return self[key]

        sample = Sample()

        for field in fields:
            sample[field] = self[field][key]

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

    def get_fields(self, fields):
        current_fields = self.fields()

        return_list = SampleList()

        for field in fields:
            if field not in current_fields:
                raise AttributeError("{} not present in SampleList. "
                                     "Valid choices are {}"
                                     .format(field, current_fields))
            return_list.add_field(field, self[field])

        return return_list

    def get_field(self, field):
        return self[field]

    def _get_data_copy(self, data):
        if isinstance(data, torch.Tensor):
            copy_ = data.clone()
        else:
            copy_ = deepcopy(data)
        return copy_

    def _get_tensor_field(self):
        return self.__dict__.get(SampleList._TENSOR_FIELD_, None)

    def _set_tensor_field(self, value):
        self.__dict__[SampleList._TENSOR_FIELD_] = value

    def get_batch_size(self):
        tensor_field = self._get_tensor_field()
        assert tensor_field is not None, "There is no tensor yet in SampleList"

        return self[tensor_field].size(0)

    def add_field(self, field, data):
        fields = self.fields()
        tensor_field = self._get_tensor_field()

        if len(fields) == 0:
            self[field] = self._get_data_copy(data)
        else:
            if isinstance(data, torch.Tensor) and \
                len(data.size()) != 0 and \
                tensor_field is not None and \
                data.size(0) != self[tensor_field].size(0):
                raise AssertionError("A tensor field to be added must "
                                     "have same size as existing tensor "
                                     "fields in SampleList. "
                                     "Passed size: {}, Required size: {}"
                                     .format(len(data), len(self[fields[0]])))
            self[field] = self._get_data_copy(data)

        if isinstance(self[field], torch.Tensor) and tensor_field is None:
            self._set_tensor_field(field)

    def to(self, device, non_blocking=True):
        fields = self.keys()
        sample_list = self.copy()
        if not isinstance(device, torch.device):
            if not isinstance(device, str):
                raise TypeError("device must be either 'str' or "
                                "'torch.device' type, {} found"
                                .format(type(device)))
            device = torch.device(device)

        for field in fields:
            if hasattr(sample_list[field], "to"):
                sample_list[field] = sample_list[field].to(
                    device, non_blocking=True
                )

        return sample_list
