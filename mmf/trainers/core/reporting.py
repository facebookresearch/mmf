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
                loss_key = report.dataset_type + "/total_loss"
                reduced_loss = sum([loss.mean() for loss in reduced_loss_dict.values()])
                if hasattr(reduced_loss, "item"):
                    reduced_loss = reduced_loss.item()

                registry.register(loss_key, reduced_loss)
                meter_update_dict.update({loss_key: reduced_loss})
                meter_update_dict.update(reduced_loss_dict)
            if hasattr(report, "metrics"):
                meter_update_dict.update(reduced_metrics_dict)
            meter.update(meter_update_dict, report.batch_size)
