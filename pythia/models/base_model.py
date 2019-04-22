# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import warnings

from torch import nn

from pythia.common.registry import registry
from pythia.common.report import Report
from pythia.modules.losses import Losses
from pythia.modules.metrics import Metrics


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.writer = registry.get("writer")

    def init_losses_and_metrics(self):
        self.loss = Losses(self.config.losses)
        self.metrics = Metrics(self.config.metrics)

    @classmethod
    def init_args(cls, parser):
        return parser

    def __call__(self, sample_list, *args, **kwargs):
        model_output = super().__call__(sample_list, *args, **kwargs)

        # Make sure theat the output from the model is a Mapping
        assert isinstance(model_output, collections.Mapping), (
            "A dict must " "be returned from the forward of the model."
        )

        if "losses" in model_output:
            warnings.warn(
                "'losses' already present in model output. "
                "No calculation will be done in base model."
            )
            assert isinstance(
                model_output["losses"], collections.Mapping
            ), "'losses' must be a dict."
        else:
            model_output["losses"] = self.loss(sample_list, model_output)

        if "metrics" in model_output:
            warnings.warn(
                "'metrics' already present in model output. "
                "No calculation will be done in base model."
            )
            assert isinstance(
                model_output["metrics"], collections.Mapping
            ), "'metrics' must be a dict."
        else:
            model_output["metrics"] = self.metrics(sample_list, model_output)

        return model_output
