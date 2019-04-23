# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import warnings

from pythia.common.registry import registry
from pythia.utils.general import get_optimizer_parameters


def build_model(config):
    model_name = config.model

    model_class = registry.get_model_class(model_name)

    if model_class is None:
        registry.get("writer").write("No model registered for name: %s" % model_name)
    model = model_class(config)

    if hasattr(model, "build"):
        model.build()
        model.init_losses_and_metrics()

    return model


def build_optimizer(model, config):
    optimizer_config = config.optimizer_attributes
    if not hasattr(optimizer_config, "type"):
        raise ValueError("Optimizer attributes must have a 'type' key "
                         "specifying the type of optimizer. "
                         "(Custom or PyTorch)")
    optimizer_type = optimizer_config.type

    if not hasattr(optimizer_config, "params"):
        warnings.warn("optimizer attributes has no params defined, "
                      "defaulting to {}.")

    params = getattr(optimizer_config, "params", {})

    if hasattr(torch.optim, optimizer_type):
        optimizer_class = getattr(torch.optim, optimizer_type)
    else:
        optimizer_class = registry.get_optimizer_class(optimizer_type)
        if optimizer_class is None:
            raise ValueError("No optimizer class of type {} present in "
                             "either torch or registered to registry")

    parameters = get_optimizer_parameters(model, config)
    optimizer = optimizer_class(parameters, **params)
    return optimizer
