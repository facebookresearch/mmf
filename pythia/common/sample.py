# Copyright (c) Facebook, Inc. and its affiliates.
"""
``Sample`` and ``SampleList`` are data structures for arbitary data returned from a
dataset. To work with Pythia, minimum requirement for datasets is to return
an object of ``Sample`` class and for models to accept an object of type `SampleList`
as an argument.

``Sample`` is used to represent an arbitary sample from dataset, while ``SampleList``
is list of Sample combined in an efficient way to be used by the model.
In simple term, ``SampleList`` is a batch of Sample but allow easy access of
attributes from ``Sample`` while taking care of properly batching things.
"""

import collections
from collections import OrderedDict
from copy import deepcopy

import torch


class Sample(OrderedDict):
    """Sample represent some arbitary data. All datasets in Pythia must
    return an object of type ``Sample``.

    Args:
        init_dict (Dict): Dictionary to init ``Sample`` class with.

    Usage::

        >>> sample = Sample({"text": torch.tensor(2)})
        >>> sample.text.zero_()
        # Custom attributes can be added to ``Sample`` after initialization
        >>> sample.context = torch.tensor(4)
    """

    def __init__(self, init_dict={}):
        super().__init__(init_dict)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def fields(self):
        """Get current attributes/fields registered under the sample.

        Returns:
            List[str]: Attributes registered under the Sample.

        """
        return list(self.keys())


class SampleList(OrderedDict):
    """``SampleList`` is used to collate a list of ``Sample`` into a batch during batch
    preparation. It can be thought of as a merger of list of Dicts into a single Dict.

    If ``Sample`` contains an attribute 'text' of size (2) and there are 10 samples in
    list, the returned ``SampleList`` will have an attribute 'text' which is a tensor
    of size (10, 2).

    Args:
        samples (type): List of ``Sample`` from which the ``SampleList`` will be created.

    Usage::

        >>> sample_list = [Sample({"text": torch.tensor(2)}), Sample({"text": torch.tensor(2)})]
        >>> sample_list.text
        torch.tensor([2, 2])
    """

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
                if (
                    isinstance(sample[field], torch.Tensor)
                    and len(sample[field].size()) != 0
                    and sample[field].size(0) != samples[0][field].size(0)
                ):
                    raise AssertionError(
                        "Fields for all samples must be equally sized. "
                        "{} is of different sizes".format(field)
                    )

                self[field][idx] = self._get_data_copy(sample[field])

            if isinstance(samples[0][field], collections.abc.Mapping):
                self[field] = SampleList(self[field])

    def _check_and_load_tuple(self, samples):
        if isinstance(samples[0], (tuple, list)) and isinstance(samples[0][0], str):
            for kv_pair in samples:
                self.add_field(kv_pair[0], kv_pair[1])
            return True
        else:
            return False

    def _check_and_load_dict(self, samples):
        if isinstance(samples, collections.abc.Mapping):
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
            raise AttributeError(
                "Key {} not found in the SampleList. "
                "Valid choices are {}".format(key, self.fields())
            )
        fields = self.keys()

        if key in fields:
            return self[key]

        sample = Sample()

        for field in fields:
            sample[field] = self[field][key]

        return sample

    def get_item_list(self, key):
        """Get ``SampleList`` of only one particular attribute that is present
        in the ``SampleList``.

        Args:
            key (str): Attribute whose ``SampleList`` will be made.

        Returns:
            SampleList: SampleList containing only the attribute value of the key
            which was passed.

        """
        sample = self[key]

        return SampleList([sample])

    def copy(self):
        """Get a copy of the current SampleList

        Returns:
            SampleList: Copy of current SampleList.

        """
        sample_list = SampleList()

        fields = self.fields()

        for field in fields:
            sample_list.add_field(field, self[field])

        return sample_list

    def fields(self):
        """Get current attributes/fields registered under the SampleList.

        Returns:
            List[str]: list of attributes of the SampleList.

        """
        return list(self.keys())

    def get_fields(self, fields):
        """Get a new ``SampleList`` generated from the current ``SampleList``
        but contains only the attributes passed in `fields` argument

        Args:
            fields (List[str]): Attributes whose ``SampleList`` will be made.

        Returns:
            SampleList: SampleList containing only the attribute values of the fields
            which were passed.

        """
        current_fields = self.fields()

        return_list = SampleList()

        for field in fields:
            if field not in current_fields:
                raise AttributeError(
                    "{} not present in SampleList. "
                    "Valid choices are {}".format(field, current_fields)
                )
            return_list.add_field(field, self[field])

        return return_list

    def get_field(self, field):
        """Get value of a particular attribute

        Args:
            field (str): Attribute whose value is to be returned.
        """
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
        """Get batch size of the current ``SampleList``. There must be a tensor
        field present in the ``SampleList`` currently.

        Returns:
            int: Size of the batch in ``SampleList``.

        """
        tensor_field = self._get_tensor_field()
        assert tensor_field is not None, "There is no tensor yet in SampleList"

        return self[tensor_field].size(0)

    def add_field(self, field, data):
        """Add an attribute ``field`` with value ``data`` to the SampleList

        Args:
            field (str): Key under which the data will be added.
            data (object): Data to be added, can be a ``torch.Tensor``, ``list``
                         or ``Sample``
        """
        fields = self.fields()
        tensor_field = self._get_tensor_field()

        if len(fields) == 0:
            self[field] = self._get_data_copy(data)
        else:
            if (
                isinstance(data, torch.Tensor)
                and len(data.size()) != 0
                and tensor_field is not None
                and data.size(0) != self[tensor_field].size(0)
            ):
                raise AssertionError(
                    "A tensor field to be added must "
                    "have same size as existing tensor "
                    "fields in SampleList. "
                    "Passed size: {}, Required size: {}".format(
                        len(data), len(self[fields[0]])
                    )
                )
            self[field] = self._get_data_copy(data)

        if isinstance(self[field], torch.Tensor) and tensor_field is None:
            self._set_tensor_field(field)

    def to(self, device, non_blocking=True):
        """Similar to ``.to`` function on a `torch.Tensor`. Moves all of the
        tensors present inside the ``SampleList`` to a particular device. If an
        attribute's value is not a tensor, it is ignored and kept as it is.

        Args:
            device (str|torch.device): Device on which the ``SampleList`` should
                                       moved.
            non_blocking (bool): Whether the move should be non_blocking. Default: True

        Returns:
            SampleList: a SampleList moved to the ``device``.

        """
        fields = self.keys()
        sample_list = self.copy()
        if not isinstance(device, torch.device):
            if not isinstance(device, str):
                raise TypeError(
                    "device must be either 'str' or "
                    "'torch.device' type, {} found".format(type(device))
                )
            device = torch.device(device)

        for field in fields:
            if hasattr(sample_list[field], "to"):
                sample_list[field] = sample_list[field].to(
                    device, non_blocking=non_blocking
                )

        return sample_list
