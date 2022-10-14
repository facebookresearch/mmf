# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import collections.abc
import copy
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from mmf.common.sample import detach_tensor, SampleList


class Report(OrderedDict):
    def __init__(
        self, batch: SampleList = None, model_output: Dict[str, Any] = None, *args
    ):
        super().__init__(self)
        if batch is None:
            return
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

    def get_batch_size(self) -> int:
        return self.batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
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

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def fields(self) -> List[str]:
        return list(self.keys())

    def apply_fn(self, fn: Callable, fields: Optional[List[str]] = None):
        """Applies a function `fn` on all items in a report. Can apply to specific
        fields if `fields` parameter is passed

        Args:
            fn (Callable): A callable to called on each item in report
            fields (List[str], optional): Use to apply on specific fields.
                Defaults to None.

        Returns:
            Report: Update report after apply fn
        """
        for key in self.keys():
            if fields is not None and isinstance(fields, (list, tuple)):
                if key not in fields:
                    continue
            self[key] = fn(self[key])
            if isinstance(self[key], collections.abc.MutableSequence):
                for idx, item in enumerate(self[key]):
                    self[key][idx] = fn(item)
            elif isinstance(self[key], dict):
                for subkey in self[key].keys():
                    self[key][subkey] = fn(self[key][subkey])
        return self

    def detach(self) -> "Report":
        """Similar to tensor.detach, detach all items in a report from their graphs.
        This is useful in clearing up memory sometimes.

        Returns:
            Report: Detached report is returned back.
        """
        return self.apply_fn(detach_tensor)

    def to(
        self,
        device: Union[torch.device, str],
        non_blocking: bool = True,
        fields: Optional[List[str]] = None,
    ):
        """Move report to a specific device defined 'device' parameter.
        This is similar to how one moves a tensor or sample_list to a device

        Args:
            device (torch.device): Device can be str defining device or torch.device
            non_blocking (bool, optional): Whether transfer should be non_blocking.
                Defaults to True.
            fields (List[str], optional): Use this is you only want to move some
                specific fields to the device instead of full report. Defaults to None.

        Raises:
            TypeError: If device type is not correct

        Returns:
            Report: Updated report is returned back
        """
        if not isinstance(device, torch.device):
            if not isinstance(device, str):
                raise TypeError(
                    "device must be either 'str' or "
                    "'torch.device' type, {} found".format(type(device))
                )
            device = torch.device(device)

        def fn(x):
            if hasattr(x, "to"):
                x = x.to(device, non_blocking=non_blocking)
            return x

        return self.apply_fn(fn, fields)

    def accumulate_tensor_fields_and_loss(
        self, report: "Report", field_list: List[str]
    ):
        for key in field_list:
            if key == "__prediction_report__":
                continue
            if key not in self.keys():
                warnings.warn(
                    f"{key} not found in report. Metrics calculation "
                    + "might not work as expected."
                )
                continue
            if isinstance(self[key], torch.Tensor):
                self[key] = torch.cat((self[key], report[key]), dim=0)
            elif isinstance(self[key], List):
                self[key].extend(report[key])

        self._accumulate_loss(report)

    def _accumulate_loss(self, report: "Report"):
        for key, value in report.losses.items():
            if key not in self.losses.keys():
                warnings.warn(
                    f"{key} not found in report. Loss calculation "
                    + "might not work as expected."
                )
                self.losses[key] = value
            if isinstance(self.losses[key], torch.Tensor):
                self.losses[key] += value

    def copy(self) -> "Report":
        """Get a copy of the current Report

        Returns:
            Report: Copy of current Report.

        """
        report = Report()

        fields = self.fields()

        for field in fields:
            report[field] = copy.deepcopy(self[field])

        return report
