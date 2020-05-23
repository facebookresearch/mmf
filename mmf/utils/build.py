# Copyright (c) Facebook, Inc. and its affiliates.
import os
import warnings

import torch
from omegaconf import OmegaConf

from mmf.common import typings as mmf_typings
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
        model.load_requirements()
        model.build()
        model.init_losses()

    return model


def build_dataset(
    dataset_key: str, config=None, dataset_type="train"
) -> mmf_typings.DatasetType:
    """Builder function for creating a dataset. If dataset_key is passed
    the dataset is created from default config of the dataset and thus is
    disable config even if it is passed. Otherwise, we use MultiDatasetLoader to
    build and return an instance of dataset based on the config

    Args:
        dataset_key (str): Key of dataset to build.
        config (DictConfig, optional): Configuration that will be used to create
            the dataset. If not passed, dataset's default config will be used.
            Defaults to {}.
        dataset_type (str, optional): Type of the dataset to build, train|val|test.
            Defaults to "train".

    Returns:
        (DatasetType): A dataset instance of type BaseDataset
    """
    from mmf.utils.configuration import load_yaml_with_defaults

    dataset_builder = registry.get_builder_class(dataset_key)
    assert dataset_builder, (
        f"Key {dataset_key} doesn't have a registered " + "dataset builder"
    )

    # If config is not provided, we take it from default one
    if not config:
        config = load_yaml_with_defaults(dataset_builder.config_path())
        config = OmegaConf.select(config, f"dataset_config.{dataset_key}")
        OmegaConf.set_struct(config, True)

    builder_instance: mmf_typings.DatasetBuilderType = dataset_builder()
    builder_instance.build_dataset(config, dataset_type)
    dataset = builder_instance.load_dataset(config, dataset_type)
    builder_instance.update_registry_for_model(config)

    return dataset


def build_dataloader_and_sampler(
    dataset_instance: mmf_typings.DatasetType, training_config: mmf_typings.DictConfig
) -> mmf_typings.DataLoaderAndSampler:
    """Builds and returns a dataloader along with its sample

    Args:
        dataset_instance (mmf_typings.DatasetType): Instance of dataset for which
            dataloader has to be created
        training_config (mmf_typings.DictConfig): Training configuration; required
            for infering params for dataloader

    Returns:
        mmf_typings.DataLoaderAndSampler: Tuple of Dataloader and Sampler instance
    """
    from mmf.common.batch_collator import BatchCollator

    num_workers = training_config.num_workers
    pin_memory = training_config.pin_memory

    other_args = {}

    other_args = _add_extra_args_for_dataloader(dataset_instance, other_args)

    loader = torch.utils.data.DataLoader(
        dataset=dataset_instance,
        pin_memory=pin_memory,
        collate_fn=BatchCollator(
            dataset_instance.dataset_name, dataset_instance.dataset_type
        ),
        num_workers=num_workers,
        **other_args,
    )

    if num_workers >= 0:
        # Suppress leaking semaphore warning
        os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

    loader.dataset_type = dataset_instance.dataset_type

    return loader, other_args.get("sampler", None)


def _add_extra_args_for_dataloader(
    dataset_instance: mmf_typings.DatasetType,
    other_args: mmf_typings.DataLoaderArgsType = None,
) -> mmf_typings.DataLoaderArgsType:
    from mmf.utils.general import get_batch_size

    if other_args is None:
        other_args = {}
    dataset_type = dataset_instance.dataset_type

    other_args["shuffle"] = False
    if dataset_type != "test":
        other_args["shuffle"] = True

    # In distributed mode, we use DistributedSampler from PyTorch
    if torch.distributed.is_initialized():
        other_args["sampler"] = torch.utils.data.DistributedSampler(
            dataset_instance, shuffle=other_args["shuffle"]
        )
        # Shuffle is mutually exclusive with sampler, let DistributedSampler
        # take care of shuffle and pop from main args
        other_args.pop("shuffle")

    other_args["batch_size"] = get_batch_size()

    return other_args


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
