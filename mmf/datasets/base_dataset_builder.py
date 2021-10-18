# Copyright (c) Facebook, Inc. and its affiliates.
"""
In MMF, for adding new datasets, dataset builder for datasets need to be
added. A new dataset builder must inherit ``BaseDatasetBuilder`` class and
implement ``load`` and ``build`` functions.

``build`` is used to build a dataset when it is not available. For e.g.
downloading the ImDBs for a dataset. In future, we plan to add a ``build``
to add dataset builder to ease setup of MMF.

``load`` is used to load a dataset from specific path. ``load`` needs to return
an instance of subclass of ``mmf.datasets.base_dataset.BaseDataset``.

See complete example for ``VQA2DatasetBuilder`` here_.

Example::

    from torch.utils.data import Dataset

    from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
    from mmf.common.registry import registry

    @registry.register_builder("my")
    class MyBuilder(BaseDatasetBuilder):
        def __init__(self):
            super().__init__("my")

        def load(self, config, dataset_type, *args, **kwargs):
            ...
            return Dataset()

        def build(self, config, dataset_type, *args, **kwargs):
            ...

.. _here: https://github.com/facebookresearch/mmf/blob/main/mmf/datasets/vqa/vqa2/builder.py
"""
import uuid
from typing import Optional

import pytorch_lightning as pl
from mmf.utils.build import build_dataloader_and_sampler
from mmf.utils.logger import log_class_usage
from omegaconf import DictConfig
from torch.utils.data import Dataset


# TODO(asg): Deprecate BaseDatasetBuilder after version release
class BaseDatasetBuilder(pl.LightningDataModule):
    """Base class for implementing dataset builders. See more information
    on top. Child class needs to implement ``build`` and ``load``.

    Args:
        dataset_name (str): Name of the dataset passed from child.
    """

    def __init__(self, dataset_name: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if dataset_name is None:
            # In case user doesn't pass it
            dataset_name = f"dataset_{uuid.uuid4().hex[:6]}"
        self.dataset_name = dataset_name
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        log_class_usage("DatasetBuilder", self.__class__)

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name):
        self._dataset_name = dataset_name

    def prepare_data(self, config, *args, **kwargs):
        """
        NOTE: The caller to this function should only call this on main process
        in a distributed settings so that downloads and build only happen
        on main process and others can just load it. Make sure to call
        synchronize afterwards to bring all processes in sync.

        Lightning automatically wraps datamodule in a way that it is only
        called on a main node, but for extra precaution as lightning
        can introduce bugs, we should always call this under main process
        with extra checks on our sides as well.
        """
        self.config = config
        self.build_dataset(config)

    def setup(self, stage: Optional[str] = None, config: Optional[DictConfig] = None):
        if config is None:
            config = self.config

        self.config = config
        self.train_dataset = self.load_dataset(config, "train")
        self.val_dataset = self.load_dataset(config, "val")
        self.test_dataset = self.load_dataset(config, "test")

    @property
    def train_dataset(self) -> Optional[Dataset]:
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, dataset: Optional[Dataset]):
        self._train_dataset = dataset

    @property
    def val_dataset(self) -> Optional[Dataset]:
        return self._val_dataset

    @val_dataset.setter
    def val_dataset(self, dataset: Optional[Dataset]):
        self._val_dataset = dataset

    @property
    def test_dataset(self) -> Optional[Dataset]:
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, dataset: Optional[Dataset]):
        self._test_dataset = dataset

    def build_dataset(self, config, dataset_type="train", *args, **kwargs):
        """
        Similar to load function, used by MMF to build a dataset for first
        time when it is not available. This internally calls 'build' function.
        Override that function in your child class.

        NOTE: The caller to this function should only call this on main process
        in a distributed settings so that downloads and build only happen
        on main process and others can just load it. Make sure to call
        synchronize afterwards to bring all processes in sync.

        Args:
            config (DictConfig): Configuration of this dataset loaded from
                                 config.
            dataset_type (str): Type of dataset, train|val|test

        .. warning::

            DO NOT OVERRIDE in child class. Instead override ``build``.
        """
        self.build(config, dataset_type, *args, **kwargs)

    def load_dataset(self, config, dataset_type="train", *args, **kwargs):
        """Main load function use by MMF. This will internally call ``load``
        function. Calls ``init_processors`` and ``try_fast_read`` on the
        dataset returned from ``load``

        Args:
            config (DictConfig): Configuration of this dataset loaded from config.
            dataset_type (str): Type of dataset, train|val|test

        Returns:
            dataset (BaseDataset): Dataset containing data to be trained on

        .. warning::

            DO NOT OVERRIDE in child class. Instead override ``load``.
        """
        dataset = self.load(config, dataset_type, *args, **kwargs)
        if dataset is not None and hasattr(dataset, "init_processors"):
            # Checking for init_processors allows us to load some datasets
            # which don't have processors and don't inherit from BaseDataset
            dataset.init_processors()
        return dataset

    def load(self, config, dataset_type="train", *args, **kwargs):
        """
        This is used to prepare the dataset and load it from a path.
        Override this method in your child dataset builder class.

        Args:
            config (DictConfig): Configuration of this dataset loaded from config.
            dataset_type (str): Type of dataset, train|val|test

        Returns:
            dataset (BaseDataset): Dataset containing data to be trained on
        """
        raise NotImplementedError(
            "This dataset builder doesn't implement a load method"
        )

    @classmethod
    def config_path(cls):
        return None

    def build(self, config, dataset_type="train", *args, **kwargs):
        """
        This is used to build a dataset first time.
        Implement this method in your child dataset builder class.

        Args:
            config (DictConfig): Configuration of this dataset loaded from
                                 config.
            dataset_type (str): Type of dataset, train|val|test
        """
        raise NotImplementedError(
            "This dataset builder doesn't implement a build method"
        )

    def build_dataloader(
        self, dataset_instance: Optional[Dataset], dataset_type: str, *args, **kwargs
    ):
        if dataset_instance is None:
            raise TypeError(
                f"dataset instance for {dataset_type} hasn't been set and is None"
            )
        dataset_instance.dataset_type = dataset_type
        dataloader, _ = build_dataloader_and_sampler(dataset_instance, self.config)
        return dataloader

    def train_dataloader(self, *args, **kwargs):
        return self.build_dataloader(self.train_dataset, "train")

    def val_dataloader(self, *args, **kwargs):
        return self.build_dataloader(self.val_dataset, "val")

    def test_dataloader(self, *args, **kwargs):
        return self.build_dataloader(self.test_dataset, "test")

    def teardown(self, *args, **kwargs) -> None:
        pass
