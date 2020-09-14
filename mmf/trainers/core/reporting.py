# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC
from typing import Any, Dict, Type

import torch
from mmf.common.meter import Meter
from mmf.common.registry import registry
from mmf.utils.distributed import reduce_dict


class TrainerReportingMixin(ABC):
    meter: Type[Meter] = Meter()

    def update_meter(
        self, report: Dict[str, Any], meter: Type[Meter] = None, eval_mode: bool = False
    ) -> None:
        if meter is None:
            meter = self.meter

        if hasattr(report, "metrics"):
            metrics_dict = report.metrics
            reduced_metrics_dict = reduce_dict(metrics_dict)

        if not eval_mode:
            loss_dict = report.losses
            reduced_loss_dict = reduce_dict(loss_dict)

        with torch.no_grad():
            # Add metrics to meter only when mode is `eval`
            meter_update_dict = {}
            if not eval_mode:
                total_loss_key = report.dataset_type + "/total_loss"
                meter_update_dict, total_loss = self.update_dict(
                    meter_update_dict, reduced_loss_dict
                )
                registry.register(total_loss_key, total_loss)
                meter_update_dict.update({total_loss_key: total_loss})

            if hasattr(report, "metrics"):
                meter_update_dict, _ = self.update_dict(
                    meter_update_dict, reduced_metrics_dict
                )

            meter.update(meter_update_dict, report.batch_size)

    def update_dict(self, meter_update_dict, values_dict):
        total_val = 0
        for key, val in values_dict.items():
            if torch.is_tensor(val):
                if val.dim() == 1:
                    val = val.mean()

            if hasattr(val, "item"):
                val = val.item()

            meter_update_dict.update({key: val})
            total_val += val

        return meter_update_dict, total_val
