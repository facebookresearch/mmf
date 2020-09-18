# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Any, Dict, List, Type

from mmf.common.registry import registry
from mmf.common.report import Report
from mmf.datasets.processors.processors import BatchProcessor, BatchProcessorConfigType


@dataclass
class ArgMaxPredictionProcessorConfig(BatchProcessorConfigType):
    # Key that will be used for id in report
    id_key: str = "id"
    # Key that will be used for result in report
    result_key: str = "answer"


@registry.register_processor("prediction.argmax")
class ArgMaxPredictionProcessor(BatchProcessor):
    """This prediction processor returns the index with maximum score for each
    id as the answer. Expects report to have scores and id keys.
    """

    def __init__(self, config: ArgMaxPredictionProcessorConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._id_key = config.get("id_key", "id")
        self._result_key = config.get("result_key", "answer")

    def __call__(self, report: Type[Report], *args, **kwargs) -> List[Dict[str, Any]]:
        answers = report.scores.argmax(dim=1)
        predictions = []
        for idx, item_id in enumerate(report.id):
            answer = answers[idx]
            predictions.append(
                {self._id_key: item_id.item(), self._result_key: answer.item()}
            )
        return predictions
