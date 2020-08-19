# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import warnings
from collections import OrderedDict

import torch


class Report(OrderedDict):
    def __init__(self, batch, model_output=None, *args):
        super().__init__(self)
        if model_output is None:
            model_output = {}
        if self._check_and_load_tuple(batch):
            return

        all_args = [batch, model_output] + [*args]
        for idx, arg in enumerate(all_args):
            if not isinstance(arg, collections.abc.Mapping):
                raise TypeError(
                    "Argument {:d}, {} must be of instance of "
                    "collections.abc.Mapping".format(idx, arg)
                )

        self.batch_size = batch.get_batch_size()
        self.warning_string = (
            "Updating forward report with key {}"
            "{}, but it already exists in {}. "
            "Please consider using a different key, "
            "as this can cause issues during loss and "
            "metric calculations."
        )

        for idx, arg in enumerate(all_args):
            for key, item in arg.items():
                if key in self and idx >= 2:
                    log = self.warning_string.format(
                        key, "", "in previous arguments to report"
                    )
                    warnings.warn(log)
                self[key] = item

    def get_batch_size(self):
        return self.batch_size

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    def _check_and_load_tuple(self, batch):
        if isinstance(batch, collections.abc.Mapping):
            return False

        if isinstance(batch[0], (tuple, list)) and isinstance(batch[0][0], str):
            for kv_pair in batch:
                self[kv_pair[0]] = kv_pair[1]
            return True
        else:
            return False

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def fields(self):
        return list(self.keys())

    def accumulate_tensor_fields(self, report, field_list):
        for key in field_list:
            if key not in self.keys():
                warnings.warn(
                    f"{key} not found in report. Metrics calculation "
                    + "might not work as expected."
                )
                continue
            if isinstance(self[key], torch.Tensor):
                self[key] = torch.cat((self[key], report[key]), dim=0)
