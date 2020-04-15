# Copyright (c) Facebook, Inc. and its affiliates.
"""
Models built on top of Pythia need to inherit ``BaseModel`` class and adhere to
some format. To create a model for MMF, follow this quick cheatsheet.

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

    from mmf.common.registry import registry
    from mmf.models.base_model import BaseModel


    @registry.register("pythia")
    class Pythia(BaseModel):
        # config is model_config from global config
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
from copy import deepcopy

from torch import nn

from mmf.common.registry import registry
from mmf.modules.losses import Losses
from mmf.modules.metrics import Metrics


class BaseModel(nn.Module):
    """For integration with Pythia's trainer, datasets and other features,
    models needs to inherit this class, call `super`, write a build function,
    write a forward function taking a ``SampleList`` as input and returning a
    dict as output and finally, register it using ``@registry.register_model``

    Args:
        config (DictConfig): ``model_config`` configuration from global config.

    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._logged_warning = {"losses_present": False, "metrics_present": False}
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
        and ``metrics`` keys. Automatically called by MMF internally after
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
    def config_path(cls):
        return None

    @classmethod
    def format_state_key(cls, key):
        """Can be implemented if something special needs to be done
        key when pretrained model is being load. This will adapt and return
        keys according to that. Useful for backwards compatibility. See
        updated load_state_dict below. For an example, see VisualBERT model's
        code.

        Args:
            key (string): key to be formatted

        Returns:
            string: formatted key
        """
        return key

    def load_state_dict(self, state_dict, *args, **kwargs):
        copied_state_dict = deepcopy(state_dict)
        for key in list(copied_state_dict.keys()):
            formatted_key = self.format_state_key(key)
            copied_state_dict[formatted_key] = copied_state_dict.pop(key)

        return super().load_state_dict(copied_state_dict, *args, **kwargs)

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
        assert isinstance(
            model_output, collections.abc.Mapping
        ), "A dict must be returned from the forward of the model."

        if "losses" in model_output:
            if not self._logged_warning["losses_present"]:
                warnings.warn(
                    "'losses' already present in model output. "
                    "No calculation will be done in base model."
                )
                self._logged_warning["losses_present"] = True

            assert isinstance(
                model_output["losses"], collections.abc.Mapping
            ), "'losses' must be a dict."
        else:
            model_output["losses"] = self.losses(sample_list, model_output)

        if "metrics" in model_output:
            if not self._logged_warning["metrics_present"]:
                warnings.warn(
                    "'metrics' already present in model output. "
                    "No calculation will be done in base model."
                )
                self._logged_warning["metrics_present"] = True

            assert isinstance(
                model_output["metrics"], collections.abc.Mapping
            ), "'metrics' must be a dict."
        else:
            model_output["metrics"] = self.metrics(sample_list, model_output)

        return model_output
