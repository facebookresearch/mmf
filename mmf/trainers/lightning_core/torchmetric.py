# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import logging
from typing import Dict, List

import torch
from mmf.common.registry import registry
from mmf.common.sample import SampleList

logger = logging.getLogger(__name__)


class LightningTorchMetrics:
    """
    A class used in LightningTrainer to compute torchmetrics
    ---
    An example to register a torchmetric:

    from mmf.common.registry import registry
    @registry.register_torchmetric("top_k_accuracy")
    class TopKAccuracy(Metric):
        def __init__(
            self,
            k: int = 1,
            ...
        ) -> None:
            ...
        def update(self, sample_list: SampleList,
            model_output: Dict[str, Tensor]) -> None:
            ...
        def compute(self) -> Tensor:
            ...

    ---
    To config the metrics in yaml config file:

    evaluation:
        torchmetrics:
        - type: top_k_accuracy
            key: top_3_overlap
            params:
            k: 3
        - type: top_k_accuracy
            key: top_1_overlap
            params:
            k: 1

    Warning: once torchmetrics are provided, regular mmf metrics will be ignored.
    """

    def __init__(self, metric_list: collections.abc.Sequence):
        if not isinstance(metric_list, collections.abc.Sequence):
            metric_list = [metric_list]
        self.metrics, self.metric_dataset_names = self._init_metrics(metric_list)

    def _init_metrics(self, metric_list: collections.abc.Sequence):
        metrics = {}
        metric_dataset_names = {}
        for metric in metric_list:
            params = {}
            dataset_names = []
            if isinstance(metric, collections.abc.Mapping):
                if "type" not in metric:
                    raise ValueError(
                        f"Metric {metric} needs to have 'type' attribute "
                        + "or should be a string"
                    )
                metric_type = key = metric.type
                params = metric.get("params", {})
                # Support cases where uses need to give custom metric name
                if "key" in metric:
                    key = metric.key

                # One key should only be used once
                if key in metrics:
                    raise RuntimeError(
                        f"Metric with type/key '{metric_type}' has been defined more "
                        + "than once in metric list."
                    )

                # a custom list of dataset where this metric will be applied
                if "datasets" in metric:
                    dataset_names = metric.datasets
                else:
                    logger.warning(
                        f"metric '{key}' will be computed on all datasets \
                        since datasets are not provided"
                    )
            else:
                if not isinstance(metric, str):
                    raise TypeError(
                        "Metric {} has inappropriate type"
                        "'dict' or 'str' allowed".format(metric)
                    )
                metric_type = key = metric

            metric_cls = registry.get_torchmetric_class(metric_type)
            if metric_cls is None:
                raise ValueError(
                    f"No metric named {metric_type} registered to registry"
                )
            metric_instance = metric_cls(**params)
            metrics[key] = metric_instance
            metric_dataset_names[key] = dataset_names
        return metrics, metric_dataset_names

    def _is_dataset_applicable(
        self, dataset_name: str, metric_dataset_names: List[str]
    ):
        return len(metric_dataset_names) == 0 or dataset_name in metric_dataset_names

    def update(
        self,
        sample_list: SampleList,
        model_output: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ):
        dataset_name = sample_list.dataset_name
        with torch.no_grad():
            for metric_name, metric in self.metrics.items():
                if not self._is_dataset_applicable(
                    dataset_name, self.metric_dataset_names.get(metric_name, [])
                ):
                    continue
                metric.update(sample_list, model_output)

    def compute(self) -> Dict[str, torch.Tensor]:
        results = {}
        for metric_name, metric in self.metrics.items():
            results[metric_name] = metric.compute()
        return results

    def reset(self) -> None:
        for _, metric in self.metrics.items():
            metric.reset()

    def get_scalar_dict(self) -> Dict[str, torch.Tensor]:
        results = self.compute()
        scalar_dict = {}
        for k, tensor_v in results.items():
            val = torch.flatten(tensor_v)
            if val.size(0) > 1:
                # non-scalar will be ignored
                continue
            scalar_dict[k] = val.item()
        return scalar_dict
