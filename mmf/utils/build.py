# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import mmf
import pytorch_lightning as pl
import torch
from mmf.common.meter import Meter
from mmf.common.registry import registry
from mmf.datasets.iteration_strategies import (
    ConstantIterationStrategy,
    IterationStrategy,
    SizeProportionalIterationStrategy,
)
from mmf.datasets.processors.processors import Processor
from mmf.utils.configuration import Configuration, get_global_config
from mmf.utils.distributed import is_dist_initialized, is_main, is_xla, synchronize
from mmf.utils.general import get_optimizer_parameters
from omegaconf import DictConfig, OmegaConf
from packaging import version


try:
    import torch_xla.core.xla_model as xm  # noqa
    import torch_xla.distributed.parallel_loader as xla_pl  # noqa
except ImportError:
    xm = None

ProcessorDict = Dict[str, Processor]
logger = logging.getLogger(__name__)


def build_config(configuration: Configuration, *args, **kwargs) -> DictConfig:
    """Builder function for config. Freezes the configuration and registers
    configuration object and config DictConfig object to registry.

    Args:
        configuration (Configuration): Configuration object that will be
            used to create the config.

    Returns:
        (DictConfig): A config which is of type omegaconf.DictConfig
    """
    configuration.freeze()
    config = configuration.get_config()
    registry.register("config", config)
    registry.register("configuration", configuration)

    return config


def build_trainer(config: DictConfig) -> Any:
    """Builder function for creating a trainer class. Trainer class name
    is picked from the config.

    Args:
        config (DictConfig): Configuration that will be used to create
            the trainer.

    Returns:
        (BaseTrainer): A trainer instance
    """
    trainer_type = config.training.trainer
    trainer_cls = registry.get_trainer_class(trainer_type)
    trainer_obj = trainer_cls(config)

    return trainer_obj


def build_lightning_model(
    config: Union[DictConfig, "mmf.models.base_model.BaseModel.Config"],
    checkpoint_path: str = None,
) -> "mmf.models.base_model.BaseModel":
    from mmf.models.base_model import BaseModel

    if not checkpoint_path:
        model = build_model(config)
        model.is_pl_enabled = True
        return model

    # If it is not an OmegaConf object, create the object
    if not isinstance(config, DictConfig) and isinstance(config, BaseModel.Config):
        config = OmegaConf.structured(config)

    model_name = config.model
    model_class = registry.get_model_class(model_name)

    if model_class is None:
        raise RuntimeError(f"No model registered for name: {model_name}")

    """ model.build is called inside on_load_checkpoint as suggested here:
    https://github.com/PyTorchLightning/pytorch-lightning/issues/5410
    """

    if is_main():
        model_class.load_requirements(model_class, config=config)
        model = model_class.load_from_checkpoint(
            checkpoint_path, config=config, strict=False
        )
        synchronize()
    else:
        synchronize()
        model = model_class.load_from_checkpoint(
            checkpoint_path, config=config, strict=False
        )

    model.init_losses()
    model.is_pl_enabled = True
    return model


def build_model(
    config: Union[DictConfig, "mmf.models.base_model.BaseModel.Config"],
) -> "mmf.models.base_model.BaseModel":
    from mmf.models.base_model import BaseModel

    # If it is not an OmegaConf object, create the object
    if not isinstance(config, DictConfig) and isinstance(config, BaseModel.Config):
        config = OmegaConf.structured(config)

    model_name = config.model
    model_class = registry.get_model_class(model_name)

    if model_class is None:
        raise RuntimeError(f"No model registered for name: {model_name}")
    model = model_class(config)

    if hasattr(model, "build"):
        """Model build involves checkpoint loading
        If the checkpoint is not available the underlying
        methods try to download it.
        Let master build the model (download the checkpoints) while
        other ranks wait for the sync message
        Once the master has downloaded the checkpoint and built the
        model it sends the sync message, completing the synchronization
        now other cores can proceed to build the model
        using already downloaded checkpoint.
        """
        if is_main():
            model_class.load_requirements(model_class, config=config)
            model.build()
            synchronize()
        else:
            synchronize()
            model.build()
        model.init_losses()
    return model


def build_dataset(
    dataset_key: str, config=None, dataset_type="train"
) -> torch.utils.data.Dataset:
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
        (torch.utils.data.Dataset): A dataset instance of type torch Dataset
    """
    from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
    from mmf.utils.configuration import load_yaml_with_defaults

    datamodule_instance = build_datamodule(dataset_key)
    # If config is not provided, we take it from default one
    if not config:
        config_path = datamodule_instance.config_path()
        if config_path is None:
            # If config path wasn't defined, send an empty config path
            # but don't force dataset to define a config
            warnings.warn(
                f"Config path not defined for {dataset_key}, "
                + "continuing with empty config"
            )
            config = OmegaConf.create()
        else:
            config = load_yaml_with_defaults(config_path)
            config = OmegaConf.select(config, f"dataset_config.{dataset_key}")
            if config is None:
                config = OmegaConf.create()
            OmegaConf.set_struct(config, True)
    elif dataset_key in config:
        # Handle Global config
        config = config[dataset_key]

    datamodule_instance.build_dataset(config)
    dataset = datamodule_instance.load_dataset(config, dataset_type)
    if hasattr(datamodule_instance, "update_registry_for_model"):
        datamodule_instance.update_registry_for_model(config)

    return dataset


# TODO: move dataset_type enum to typings
def build_datasets(
    dataset_list: List[str], dataset_config: DictConfig, dataset_type="train"
) -> List[torch.utils.data.Dataset]:
    datasets = []
    for dataset in dataset_list:
        if dataset in dataset_config:
            dataset_config = dataset_config[dataset]
        else:
            warnings.warn(
                f"Dataset {dataset} is missing from dataset_config"
                + " in config. Proceeding with empty config."
            )
            dataset_config = OmegaConf.create()

        dataset_instance = build_dataset(dataset, dataset_config, dataset_type)
        if dataset_instance is None:
            continue
        datasets.append(dataset_instance)

    return datasets


def build_datamodule(dataset_key) -> pl.LightningDataModule:
    dataset_builder = registry.get_builder_class(dataset_key)
    assert dataset_builder, (
        f"Key {dataset_key} doesn't have a registered " + "dataset builder"
    )
    builder_instance: pl.LightningDataModule = dataset_builder()
    return builder_instance


def build_multiple_datamodules(
    dataset_list: List[str], all_dataset_config: DictConfig
) -> Dict[str, pl.LightningDataModule]:
    datamodules: Dict[str, pl.LightningDataModule] = {}
    for dataset in dataset_list:
        datamodule_instance = build_datamodule(dataset)
        if dataset in all_dataset_config:
            dataset_config = all_dataset_config[dataset]
        else:
            warnings.warn(
                f"Dataset {dataset} is missing from dataset_config"
                + " in config. Proceeding with empty config."
            )
            dataset_config = OmegaConf.create()

        if is_main():
            datamodule_instance.prepare_data(dataset_config)

        synchronize()
        datamodule_instance.setup(config=dataset_config)
        if hasattr(datamodule_instance, "update_registry_for_model"):
            datamodule_instance.update_registry_for_model(dataset_config)
        datamodules[dataset] = datamodule_instance
    return datamodules


def build_dataloader_and_sampler(
    dataset_instance: torch.utils.data.Dataset, datamodule_config: DictConfig
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.Sampler]]:
    """Builds and returns a dataloader along with its sample

    Args:
        dataset_instance (torch.utils.data.Dataset): Instance of dataset for which
            dataloader has to be created
        datamodule_config (omegaconf.DictConfig): Datamodule configuration; required
            for infering params for dataloader

    Returns:
        Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.Sampler]]:
            Tuple of Dataloader and Sampler instance
    """
    from mmf.common.batch_collator import BatchCollator

    training_config = get_global_config("training")
    # Support params coming in from dataloader params
    other_args = {
        "num_workers": datamodule_config.get(
            "num_workers", training_config.get("num_workers", 4)
        ),
        "pin_memory": datamodule_config.get(
            "pin_memory", training_config.get("pin_memory", False)
        ),
        "shuffle": datamodule_config.get("shuffle", None),
        "batch_size": datamodule_config.get("batch_size", None),
    }
    if version.parse(torch.__version__) >= version.parse("1.8"):
        # only use persistent workers in PyTorch 1.8 or higher
        # (PyTorch 1.7 also has this option but doesn't support it correctly due to
        # https://github.com/pytorch/pytorch/issues/48370)
        other_args["persistent_workers"] = (
            datamodule_config.get(
                "persistent_workers", training_config.get("persistent_workers", True)
            ),
        )
        if other_args["persistent_workers"] and other_args["num_workers"] == 0:
            logger.warning(
                "persistent_workers cannot be used together with num_workers == 0; "
                "setting persistent_workers to False"
            )
            other_args["persistent_workers"] = False

    # IterableDataset returns batches directly, so no need to add Sampler
    # or batch size as user is expected to control those. This is a fine
    # assumption for now to not support single item based IterableDataset
    # as it will add unnecessary complexity and config parameters
    # to the codebase
    if not isinstance(dataset_instance, torch.utils.data.IterableDataset):
        other_args = _add_extra_args_for_dataloader(dataset_instance, other_args)
    else:
        other_args.pop("shuffle")

    # Set drop_last=True when using XLA to have constant batch size.
    # In this case we also need to set drop_last=True in DistributedSampler.
    loader = torch.utils.data.DataLoader(
        dataset=dataset_instance,
        collate_fn=BatchCollator(
            dataset_instance.dataset_name, dataset_instance.dataset_type
        ),
        drop_last=is_xla(),  # see also MultiDatasetLoader.__len__
        **other_args,
    )

    if is_xla():
        device = xm.xla_device()
        loader = xla_pl.MpDeviceLoader(loader, device)

    if other_args["num_workers"] >= 0:
        # Suppress leaking semaphore warning
        os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

    loader.dataset_type = dataset_instance.dataset_type

    return loader, other_args.get("sampler", None)


def build_test_reporter(
    datamodules: List[pl.LightningDataModule],
    config: DictConfig = None,
    dataset_type: str = "train",
):
    test_reporter_key = "default"
    if config:
        test_reporter_key = config.get("type", "default")
    test_reporter_class = registry.get_test_rerporter_class(test_reporter_key)
    assert (
        test_reporter_class
    ), f"Key {test_reporter_key} doesn't have a registered test_reporter class"

    if not config:
        warnings.warn(
            f"Config not provided for {test_reporter_key}, test_reporter"
            + "continuing with empty config"
        )
        params_config = OmegaConf.create()
    else:
        params_config = config.params

    return test_reporter_class(datamodules, params_config, dataset_type)


def _add_extra_args_for_dataloader(
    dataset_instance: torch.utils.data.Dataset, other_args: Dict[str, Any] = None
) -> Dict[str, Any]:
    from mmf.utils.general import get_batch_size

    dataset_type = dataset_instance.dataset_type

    if other_args["shuffle"] is None:
        other_args["shuffle"] = False
        if dataset_type != "test":
            other_args["shuffle"] = True

    # In distributed mode, we use DistributedSampler from PyTorch
    if is_dist_initialized():
        other_args["sampler"] = torch.utils.data.DistributedSampler(
            dataset_instance, shuffle=other_args["shuffle"]
        )
        # Shuffle is mutually exclusive with sampler, let DistributedSampler
        # take care of shuffle and pop from main args
        other_args.pop("shuffle")

    if is_xla():
        other_args["sampler"] = torch.utils.data.DistributedSampler(
            dataset_instance,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=other_args["shuffle"],
            drop_last=True,
        )
        other_args.pop("shuffle")

    if other_args["batch_size"] is None:
        other_args["batch_size"] = get_batch_size()

    return other_args


def build_optimizer(model, config):
    optimizer_config = config.optimizer
    if "type" not in optimizer_config:
        raise ValueError(
            "Optimizer attributes must have a 'type' key "
            "specifying the type of optimizer. "
            "(Custom or PyTorch, e.g. 'adam_w' or 'SGD')"
        )
    optimizer_type = optimizer_config.type

    if "params" not in optimizer_config:
        warnings.warn("optimizer attributes has no params defined, defaulting to {}.")

    params = optimizer_config.get("params", {})

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

    if optimizer_config.get("enable_state_sharding", False):
        # TODO(vedanuj): Remove once OSS is moved to PT upstream
        try:
            from fairscale.optim.oss import OSS
        except ImportError:
            print(
                "Optimizer state sharding requires fairscale. "
                + "Install using pip install fairscale."
            )
            raise

        assert (
            is_dist_initialized()
        ), "Optimizer state sharding can only be used in distributed mode."

        is_fp16 = config.get("training", {}).get("fp16", False)
        optimizer = OSS(
            params=parameters, optim=optimizer_class, broadcast_fp16=is_fp16, **params
        )
    else:
        optimizer = optimizer_class(parameters, **params)
    return optimizer


def build_lightning_optimizers(model, config):
    optimizer = build_optimizer(model, config)

    if config.training.lr_scheduler:
        lr_scheduler = build_scheduler(optimizer, config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
    else:
        return optimizer


def build_scheduler(optimizer, config):
    scheduler_config = config.get("scheduler", {})

    if "type" not in scheduler_config:
        warnings.warn(
            "No type for scheduler specified even though lr_scheduler is True, "
            "setting default to 'Pythia'"
        )
    scheduler_type = scheduler_config.get("type", "pythia")

    if "params" not in scheduler_config:
        warnings.warn("scheduler attributes has no params defined, defaulting to {}.")
    params = scheduler_config.get("params", {})
    scheduler_class = registry.get_scheduler_class(scheduler_type)
    scheduler = scheduler_class(optimizer, **params)

    return scheduler


def build_classifier_layer(config, *args, **kwargs):
    from mmf.modules.layers import ClassifierLayer

    classifier = ClassifierLayer(config.type, *args, **config.params, **kwargs)
    return classifier.module


def build_text_encoder(config, *args, **kwargs):
    """Deprecated, please do not use"""
    try:
        from mmf.modules.fb.encoders import TextEncoderFactory
    except ImportError:
        from mmf.modules.encoders import TextEncoderFactory

    text_encoder = TextEncoderFactory(config, *args, **kwargs)
    return text_encoder.module


def build_image_encoder(config, direct_features=False, **kwargs):
    """Deprecated, please do not use"""
    from mmf.modules.encoders import ImageEncoderFactory, ImageFeatureEncoderFactory

    if direct_features:
        module = ImageFeatureEncoderFactory(config)
    else:
        module = ImageEncoderFactory(config)
    return module.module


def build_encoder(config: Union[DictConfig, "mmf.modules.encoders.Encoder.Config"]):
    from mmf.modules.encoders import Encoder

    # If it is not an OmegaConf object, create the object
    if not isinstance(config, DictConfig) and isinstance(config, Encoder.Config):
        config = OmegaConf.structured(config)

    if "type" in config:
        # Support config initialization in form of
        # encoder:
        #   type: identity # noqa
        #   params:
        #       in_dim: 256
        name = config.type
        if isinstance(name, Enum):
            name = name.value
        params = config.get("params", None)
    else:
        # Structured Config support
        name = config.name
        params = config

    encoder_cls = registry.get_encoder_class(name)

    # If params were not passed, try generating them from encoder
    # class's default config
    if params is None:
        params = OmegaConf.structured(getattr(encoder_cls, "Config", {}))

    return encoder_cls(params)


def build_processors(
    processors_config: DictConfig, registry_key: str = None, *args, **kwargs
) -> ProcessorDict:
    """Given a processor config, builds the processors present and returns back
    a dict containing processors mapped to keys as per the config

    Args:
        processors_config (omegaconf.DictConfig): OmegaConf DictConfig describing
            the parameters and type of each processor passed here

        registry_key (str, optional): If passed, function would look into registry for
            this particular key and return it back. .format with processor_key will
            be called on this string. Defaults to None.

    Returns:
        ProcessorDict: Dictionary containing key to
            processor mapping
    """
    from mmf.datasets.processors.processors import Processor

    processor_dict = {}

    for processor_key, processor_params in processors_config.items():
        if not processor_params:
            continue

        processor_instance = None
        if registry_key is not None:
            full_key = registry_key.format(processor_key)
            processor_instance = registry.get(full_key, no_warning=True)

        if processor_instance is None:
            processor_instance = Processor(processor_params, *args, **kwargs)
            # We don't register back here as in case of hub interface, we
            # want the processors to be instantiate every time. BaseDataset
            # can register at its own end
        processor_dict[processor_key] = processor_instance

    return processor_dict


def build_iteration_strategy(
    config: DictConfig,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    *args,
    **kwargs,
) -> IterationStrategy:
    if not config.get("enabled", True):
        return ConstantIterationStrategy.from_params(dataloaders, *args, **kwargs)
    else:
        assert (
            "type" in config
        ), "multitasking config must define 'type' attribute if enabled"
        # This assumes all dataloaders will have same dataset type
        iteration_strategy_class = registry.get_iteration_strategy_class(config.type)
        config = config.get("params", {})
        # val and test splits won't be affected as test reporter iterates
        # over the datasets one by one without using any iteration strategy
        return iteration_strategy_class(config, dataloaders, *args, **kwargs)


def build_meters(run_type: str) -> List[Meter]:
    train_meter, val_meter, test_meter = None, None, None
    if "train" in run_type:
        train_meter = Meter()
        # val_meter used for validation after training loop
        val_meter = Meter()
    elif "val" in run_type or "inference" in run_type:
        val_meter = Meter()

    if "test" in run_type:
        test_meter = Meter()

    return train_meter, val_meter, test_meter
