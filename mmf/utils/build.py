# Copyright (c) Facebook, Inc. and its affiliates.
import warnings

import torch

from mmf.common.registry import registry
from mmf.utils.general import get_optimizer_parameters


def build_trainer(configuration, *rest, **kwargs):
    configuration.freeze()

    config = configuration.get_config()
    registry.register("config", config)
    registry.register("configuration", configuration)

    trainer_type = config.training.trainer
    trainer_cls = registry.get_trainer_class(trainer_type)
    trainer_obj = trainer_cls(configuration)

    # Set args as an attribute for future use
    trainer_obj.args = configuration.args

    return trainer_obj


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
    optimizer_config = config.optimizer
    if not hasattr(optimizer_config, "type"):
        raise ValueError(
            "Optimizer attributes must have a 'type' key "
            "specifying the type of optimizer. "
            "(Custom or PyTorch)"
        )
    optimizer_type = optimizer_config.type

    if not hasattr(optimizer_config, "params"):
        warnings.warn("optimizer attributes has no params defined, defaulting to {}.")

    params = getattr(optimizer_config, "params", {})

    if hasattr(torch.optim, optimizer_type):
        optimizer_class = getattr(torch.optim, optimizer_type)
    else:
        optimizer_class = registry.get_optimizer_class(optimizer_type)
        if optimizer_class is None:
            raise ValueError(
                "No optimizer class of type {} present in "
                "either torch or registered to registry"
            )

    parameters = get_optimizer_parameters(model, config)
    optimizer = optimizer_class(parameters, **params)
    return optimizer


def build_scheduler(optimizer, config):
    scheduler_config = config.get("scheduler", {})

    if not hasattr(scheduler_config, "type"):
        warnings.warn(
            "No type for scheduler specified even though lr_scheduler is True, "
            "setting default to 'Pythia'"
        )
    scheduler_type = getattr(scheduler_config, "type", "pythia")

    if not hasattr(scheduler_config, "params"):
        warnings.warn("scheduler attributes has no params defined, defaulting to {}.")
    params = getattr(scheduler_config, "params", {})
    scheduler_class = registry.get_scheduler_class(scheduler_type)
    scheduler = scheduler_class(optimizer, **params)

    return scheduler


def build_classifier_layer(config, *args, **kwargs):
    from mmf.modules.layers import ClassifierLayer

    classifier = ClassifierLayer(config.type, *args, **config.params, **kwargs)
    return classifier.module


def build_text_encoder(config, *args, **kwargs):
    from mmf.modules.encoders import TextEncoder

    text_encoder = TextEncoder(config, *args, **kwargs)
    return text_encoder.module


def build_image_encoder(config, direct_features=False, **kwargs):
    from mmf.modules.encoders import ImageFeatureEncoder, ImageEncoder

    if direct_features:
        module = ImageFeatureEncoder(config.type, **config.params)
    else:
        module = ImageEncoder(config)
    return module.module
