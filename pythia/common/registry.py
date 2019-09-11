# Copyright (c) Facebook, Inc. and its affiliates.
"""
Registry is central source of truth in Pythia. Inspired from Redux's
concept of global store, Registry maintains mappings of various information
to unique keys. Special functions in registry can be used as decorators to
register different kind of classes.

Import the global registry object using

``from pythia.common.registry import registry``

Various decorators for registry different kind of classes with unique keys

- Register a trainer: ``@registry.register_trainer``
- Register a dataset builder: ``@registry.register_builder``
- Register a metric: ``@registry.register_metric``
- Register a loss: ``@registry.register_loss``
- Register a model: ``@registry.register_model``
- Register a processor: ``@registry.register_processor``
- Register a optimizer: ``@registry.register_optimizer``
- Register a scheduler: ``@registry.register_scheduler``
"""


class Registry:
    r"""Class for registry object which acts as central source of truth
    for Pythia
    """
    mapping = {
        # Mappings of builder name to their respective classes
        # Use `registry.register_builder` to register a builder class with a specific name
        # Further, use the name with the class is registered in the
        # command line or configuration to load that specific dataset
        "builder_name_mapping": {},
        # Similar to the builder_name_mapping above except that this
        # one is used to keep a mapping for dataset to its trainer class.
        "trainer_name_mapping": {},
        "model_name_mapping": {},
        "metric_name_mapping": {},
        "loss_name_mapping": {},
        "optimizer_name_mapping": {},
        "scheduler_name_mapping": {},
        "processor_name_mapping": {},
        "state": {}
    }

    @classmethod
    def register_trainer(cls, name):
        r"""Register a trainer to registry with key 'name'

        Args:
            name: Key with which the trainer will be registered.

        Usage::

            from pythia.common.registry import registry
            from pythia.trainers.custom_trainer import CustomTrainer


            @registry.register_trainer("custom_trainer")
            class CustomTrainer():
                ...

        """
        def wrap(trainer_cls):
            cls.mapping["trainer_name_mapping"][name] = trainer_cls
            return trainer_cls
        return wrap

    @classmethod
    def register_builder(cls, name):
        r"""Register a dataset builder to registry with key 'name'

        Args:
            name: Key with which the metric will be registered.

        Usage::

            from pythia.common.registry import registry
            from pythia.datasets.base_dataset_builder import BaseDatasetBuilder


            @registry.register_builder("vqa2")
            class VQA2Builder(BaseDatasetBuilder):
                ...

        """

        def wrap(builder_cls):
            from pythia.datasets.base_dataset_builder import BaseDatasetBuilder

            assert issubclass(
                builder_cls, BaseDatasetBuilder
            ), "All builders must inherit BaseDatasetBuilder class"
            cls.mapping["builder_name_mapping"][name] = builder_cls
            return builder_cls

        return wrap

    @classmethod
    def register_metric(cls, name):
        r"""Register a metric to registry with key 'name'

        Args:
            name: Key with which the metric will be registered.

        Usage::

            from pythia.common.registry import registry
            from pythia.modules.metrics import BaseMetric


            @registry.register_metric("r@1")
            class RecallAt1(BaseMetric):
                ...

        """

        def wrap(func):
            from pythia.modules.metrics import BaseMetric

            assert issubclass(
                func, BaseMetric
            ), "All Metric must inherit BaseMetric class"
            cls.mapping["metric_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_loss(cls, name):
        r"""Register a loss to registry with key 'name'

        Args:
            name: Key with which the loss will be registered.

        Usage::

            from pythia.common.registry import registry
            from torch import nn

            @registry.register_task("logit_bce")
            class LogitBCE(nn.Module):
                ...

        """

        def wrap(func):
            from torch import nn

            assert issubclass(
                func, nn.Module
            ), "All loss must inherit torch.nn.Module class"
            cls.mapping["loss_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls, name):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the model will be registered.

        Usage::

            from pythia.common.registry import registry
            from pythia.models.base_model import BaseModel

            @registry.register_task("pythia")
            class Pythia(BaseModel):
                ...
        """

        def wrap(func):
            from pythia.models.base_model import BaseModel

            assert issubclass(
                func, BaseModel
            ), "All models must inherit BaseModel class"
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_processor(cls, name):
        r"""Register a processor to registry with key 'name'

        Args:
            name: Key with which the processor will be registered.

        Usage::

            from pythia.common.registry import registry
            from pythia.datasets.processors import BaseProcessor

            @registry.register_task("glove")
            class GloVe(BaseProcessor):
                ...

        """

        def wrap(func):
            from pythia.datasets.processors import BaseProcessor

            assert issubclass(
                func, BaseProcessor
            ), "All Processor classes must inherit BaseProcessor class"
            cls.mapping["processor_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_optimizer(cls, name):
        def wrap(func):
            cls.mapping["optimizer_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_scheduler(cls, name):
        def wrap(func):
            cls.mapping["scheduler_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from pythia.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def get_trainer_class(cls, name):
        return cls.mapping["trainer_name_mapping"].get(name, None)

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping["processor_name_mapping"].get(name, None)

    @classmethod
    def get_metric_class(cls, name):
        return cls.mapping["metric_name_mapping"].get(name, None)

    @classmethod
    def get_loss_class(cls, name):
        return cls.mapping["loss_name_mapping"].get(name, None)

    @classmethod
    def get_optimizer_class(cls, name):
        return cls.mapping["optimizer_name_mapping"].get(name, None)

    @classmethod
    def get_scheduler_class(cls, name):
        return cls.mapping["scheduler_name_mapping"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retreived.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for Pythia's
                               internal operations. Default: False
        Usage::

            from pythia.common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].write(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from pythia.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()
