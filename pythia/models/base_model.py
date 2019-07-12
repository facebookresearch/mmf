# Copyright (c) Facebook, Inc. and its affiliates.
"""
Models built on top of Pythia need to inherit ``BaseModel`` class and adhere to
some format. To create a model for Pythia, follow this quick cheatsheet.

1. Inherit ``BaseModel`` class, make sure to call ``super().__init__()`` in your
   class's ``__init__`` function.
2. Implement `build` function for your model. If you build everything in ``__init__``,
   you can just return in this function.
3. Write a `forward` function which takes in a ``SampleList`` as an argument and
   returns a dict.
4. Register using ``@registry.register_model("key")`` decorator on top of the
   class.

If you are doing logits based predictions, the dict you return from your model
should contain a `scores` field. Losses and Metrics are automatically
calculated by the ``BaseModel`` class and added to this dict if not present.

Example::

    import torch

    from pythia.common.registry import registry
    from pythia.models.base_model import BaseModel


    @registry.register("pythia")
    class Pythia(BaseModel):
        # config is model_attributes from global config
        def __init__(self, config):
            super().__init__(config)

        def build(self):
            ....

        def forward(self, sample_list):
            scores = torch.rand(sample_list.get_batch_size(), 3127)
            return {"scores": scores}
"""


import collections
import warnings

from torch import nn

from pythia.common.registry import registry
from pythia.common.report import Report
from pythia.modules.losses import Losses
from pythia.modules.metrics import Metrics


class BaseModel(nn.Module):
    """For integration with Pythia's trainer, datasets and other feautures,
    models needs to inherit this class, call `super`, write a build function,
    write a forward function taking a ``SampleList`` as input and returning a
    dict as output and finally, register it using ``@registry.register_model``

    Args:
        config (ConfigNode): ``model_attributes`` configuration from global config.

    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.writer = registry.get("writer")

    def build(self):
        """Function to be implemented by the child class, in case they need to
        build their model separately than ``__init__``. All model related
        downloads should also happen here.
        """
        raise NotImplementedError(
            "Build method not implemented in the child model class."
        )

    def init_losses_and_metrics(self):
        """Initializes loss and metrics for the model based ``losses`` key
        and ``metrics`` keys. Automatically called by Pythia internally after
        building the model.
        """
        losses = self.config.get("losses", [])
        metrics = self.config.get("metrics", [])
        if len(losses) == 0:
            warnings.warn(
                "No losses are defined in model configuration. You are expected "
                "to return loss in your return dict from forward."
            )

        if len(metrics) == 0:
            warnings.warn(
                "No metrics are defined in model configuration. You are expected "
                "to return metrics in your return dict from forward."
            )
        self.losses = Losses(losses)
        self.metrics = Metrics(metrics)

    @classmethod
    def init_args(cls, parser):
        return parser

    def forward(self, sample_list, *args, **kwargs):
        """To be implemented by child class. Takes in a ``SampleList`` and
        returns back a dict.

        Args:
            sample_list (SampleList): SampleList returned by the DataLoader for
            current iteration

        Returns:
            Dict: Dict containing scores object.

        """
        raise NotImplementedError(
            "Forward of the child model class needs to be implemented."
        )

    def __call__(self, sample_list, *args, **kwargs):
        model_output = super().__call__(sample_list, *args, **kwargs)

        # Make sure theat the output from the model is a Mapping
        assert isinstance(model_output, collections.abc.Mapping), (
            "A dict must be returned from the forward of the model."
        )

        if "losses" in model_output:
            warnings.warn(
                "'losses' already present in model output. "
                "No calculation will be done in base model."
            )
            assert isinstance(
                model_output["losses"], collections.abc.Mapping
            ), "'losses' must be a dict."
        else:
            model_output["losses"] = self.losses(sample_list, model_output)

        if "metrics" in model_output:
            warnings.warn(
                "'metrics' already present in model output. "
                "No calculation will be done in base model."
            )
            assert isinstance(
                model_output["metrics"], collections.abc.Mapping
            ), "'metrics' must be a dict."
        else:
            model_output["metrics"] = self.metrics(sample_list, model_output)

        return model_output
