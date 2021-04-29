# Copyright (c) Facebook, Inc. and its affiliates.
# Inspired from maskrcnn benchmark
from collections import defaultdict, deque

import torch
from mmf.common.registry import registry
from mmf.utils.distributed import reduce_dict
from mmf.utils.general import scalarize_dict_values


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.deque = deque(maxlen=self.window_size)
        self.averaged_value_deque = deque(maxlen=self.window_size)
        self.batch_sizes = deque(maxlen=self.window_size)
        self.total_samples = 0
        self.total = 0.0
        self.count = 0

    def update(self, value, batch_size):
        self.deque.append(value * batch_size)
        self.averaged_value_deque.append(value)
        self.batch_sizes.append(batch_size)

        self.count += 1
        self.total_samples += batch_size
        self.total += value * batch_size

    @property
    def median(self):
        d = torch.tensor(list(self.averaged_value_deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        s = torch.tensor(list(self.batch_sizes))
        return d.sum().item() / s.sum().item()

    @property
    def global_avg(self):
        return self.total / self.total_samples

    def get_latest(self):
        return self.averaged_value_deque[-1]


class Meter:
    def __init__(self, delimiter=", "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update_from_report(self, report, should_update_loss=True):
        """
        this method updates the provided meter with report info.
        this method by default handles reducing metrics.

        Args:
            report (Report): report object which content is used to populate
            the current meter

        Usage::

        >>> meter = Meter()
        >>> report = Report(prepared_batch, model_output)
        >>> meter.update_from_report(report)
        """
        if hasattr(report, "metrics"):
            metrics_dict = report.metrics
            reduced_metrics_dict = reduce_dict(metrics_dict)

        if should_update_loss:
            loss_dict = report.losses
            reduced_loss_dict = reduce_dict(loss_dict)

        with torch.no_grad():
            meter_update_dict = {}
            if should_update_loss:
                meter_update_dict = scalarize_dict_values(reduced_loss_dict)
                total_loss_key = report.dataset_type + "/total_loss"
                total_loss = sum(meter_update_dict.values())
                registry.register(total_loss_key, total_loss)
                meter_update_dict.update({total_loss_key: total_loss})

            if hasattr(report, "metrics"):
                metrics_dict = scalarize_dict_values(reduced_metrics_dict)
                meter_update_dict.update(**metrics_dict)

            self._update(meter_update_dict, report.batch_size)

    def _update(self, update_dict, batch_size):
        scalarized = scalarize_dict_values(update_dict)
        for k, v in scalarized.items():
            # Skipping .item() call
            # __format__() for tensor has .item
            # Therefore it will implicitly get called when needed
            self.meters[k].update(v, batch_size)

    def update_from_meter(self, meter):
        for key, value in meter.meters.items():
            assert isinstance(value, SmoothedValue)
            self.meters[key] = value

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def get_scalar_dict(self):
        scalar_dict = {}
        for k, v in self.meters.items():
            scalar_dict[k] = v.get_latest()

        return scalar_dict

    def get_log_dict(self):
        log_dict = {}
        for k, v in self.meters.items():
            if "train" in k:
                log_dict[k] = f"{v.median:.4f}"
                log_dict[f"{k}/avg"] = f"{v.global_avg:.4f}"
            else:
                log_dict[k] = f"{v.global_avg:.4f}"
        return log_dict

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if "train" in name:
                loss_str.append(f"{name}: {meter.median:.4f} ({meter.global_avg:.4f})")
            else:
                # In case of val print global avg
                loss_str.append(f"{name}: {meter.global_avg:.4f}")

        return self.delimiter.join(loss_str)

    def reset(self):
        del self.meters
        self.meters = defaultdict(SmoothedValue)
