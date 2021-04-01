# Copyright (c) Facebook, Inc. and its affiliates.
"""
``Sample`` and ``SampleList`` are data structures for arbitrary data returned from a
dataset. To work with MMF, minimum requirement for datasets is to return
an object of ``Sample`` class and for models to accept an object of type `SampleList`
as an argument.

``Sample`` is used to represent an arbitrary sample from dataset, while ``SampleList``
is list of Sample combined in an efficient way to be used by the model.
In simple term, ``SampleList`` is a batch of Sample but allow easy access of
attributes from ``Sample`` while taking care of properly batching things.
"""

import collections
import warnings
from collections import OrderedDict
from typing import Any, Dict, Union

import torch


class Sample(OrderedDict):
    """Sample represent some arbitrary data. All datasets in MMF must
    return an object of type ``Sample``.

    Args:
        init_dict (Dict): Dictionary to init ``Sample`` class with.

    Usage::

        >>> sample = Sample({"text": torch.tensor(2)})
        >>> sample.text.zero_()
        # Custom attributes can be added to ``Sample`` after initialization
        >>> sample.context = torch.tensor(4)
    """

    def __init__(self, init_dict=None):
        if init_dict is None:
            init_dict = {}
        super().__init__(init_dict)

    def __setattr__(self, key, value):
        if isinstance(value, collections.abc.Mapping):
            value = Sample(value)
        self[key] = value

    def __setitem__(self, key, value):
        if isinstance(value, collections.abc.Mapping):
            value = Sample(value)
        super().__setitem__(key, value)

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
        samples (type): List of ``Sample`` from which the ``SampleList``
                        will be created.

    Usage::

        >>> sample_list = [
                Sample({"text": torch.tensor(2)}),
                Sample({"text": torch.tensor(2)})
            ]
        >>> sample_list.text
        torch.tensor([2, 2])
    """

    _TENSOR_FIELD_ = "_tensor_field"

    def __init__(self, samples=None):
        super().__init__(self)
        if samples is None:
            samples = []

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

    def __setattr__(self, key, value):
        self[key] = value

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

    def get_device(self):
        field_tensor = self._get_tensor_field()
        assert (
            field_tensor is not None
        ), f"No tensor field in sample list, available keys: {self.fields()}"
        return self[field_tensor].device

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
        # if isinstance(data, torch.Tensor):
        #     copy_ = data.clone()
        # else:
        #     copy_ = deepcopy(data)
        # return copy_
        return data

    def _get_tensor_field(self):
        return self.__dict__.get(SampleList._TENSOR_FIELD_, None)

    def _set_tensor_field(self, value):
        self.__dict__[SampleList._TENSOR_FIELD_] = value

    def get_batch_size(self):
        """Get batch size of the current ``SampleList``. There must be a tensor
        be a tensor present inside sample list to use this function.
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

        if (
            len(fields) != 0
            and isinstance(data, torch.Tensor)
            and len(data.size()) != 0
            and tensor_field is not None
            and data.size(0) != self[tensor_field].size(0)
        ):
            raise AssertionError(
                "A tensor field to be added must "
                "have same size as existing tensor "
                "fields in SampleList. "
                "Passed size: {}, Required size: {}".format(
                    len(data), len(self[tensor_field])
                )
            )

        if isinstance(data, collections.abc.Mapping):
            self[field] = SampleList(data)
        else:
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

    def pin_memory(self):
        """In custom batch object, we need to define pin_memory function so that
        PyTorch can actually apply pinning. This function just individually pins
        all of the tensor fields
        """
        fields = self.keys()

        for field in fields:
            if hasattr(self[field], "pin_memory"):
                # This will also handle nested sample list recursively
                self[field] = self[field].pin_memory()

        return self

    def detach(self):
        fields = self.keys()

        for field in fields:
            self[field] = detach_tensor(self[field])

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Converts a sample list to dict, this is useful for TorchScript and for
        other internal API unification efforts.

        Returns:
            Dict[str, Any]: A dict representation of current sample list
        """
        sample_dict = {}
        fields = self.keys()

        for field in fields:
            # Handle nested sample list recursively
            if hasattr(self[field], "to_dict"):
                sample_dict[field] = self[field].to_dict()
            else:
                sample_dict[field] = self[field]

        return sample_dict


def convert_batch_to_sample_list(
    batch: Union[SampleList, Dict[str, Any]]
) -> SampleList:
    # Create and return sample list with proper name
    # and type set if it is already not a sample list
    # (case of batched iterators)
    sample_list = batch
    if (
        # Check if batch is a list before checking batch[0]
        # or len as sometimes batch is already SampleList
        isinstance(batch, list)
        and len(batch) == 1
        and isinstance(batch[0], SampleList)
    ):
        sample_list = batch[0]
    elif not isinstance(batch, SampleList):
        sample_list = SampleList(batch)

    if sample_list._get_tensor_field() is None:
        sample_list = SampleList(sample_list.to_dict())

    return sample_list


device_type = Union[str, torch.device]


def to_device(
    sample_list: Union[SampleList, Dict[str, Any]], device: device_type = "cuda"
) -> SampleList:
    if isinstance(sample_list, collections.Mapping):
        sample_list = convert_batch_to_sample_list(sample_list)
    # to_device is specifically for SampleList
    # if user is passing something custom built
    if not isinstance(sample_list, SampleList):
        warnings.warn(
            "You are not returning SampleList/Sample from your dataset. "
            "MMF expects you to move your tensors to cuda yourself."
        )
        return sample_list

    if isinstance(device, str):
        device = torch.device(device)

    # default value of device_type is cuda
    # Other device types such as xla can also be passed.
    # Fall back to cpu only happens when device_type
    # is set to cuda but cuda is not available.
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "Selected device is cuda, but it is NOT available!!! Falling back on cpu."
        )
        device = torch.device("cpu")

    if sample_list.get_device() != device:
        sample_list = sample_list.to(device)
    return sample_list


def detach_tensor(tensor: Any) -> Any:
    """Detaches any element passed which has a `.detach` function defined.
    Currently, in MMF can be SampleList, Report or a tensor.

    Args:
        tensor (Any): Item to be detached

    Returns:
        Any: Detached element
    """
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    return tensor
