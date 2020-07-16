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

.. _here: https://github.com/facebookresearch/mmf/blob/master/mmf/datasets/vqa/vqa2/builder.py
"""

from mmf.utils.distributed import is_master, synchronize


class BaseDatasetBuilder:
    """Base class for implementing dataset builders. See more information
    on top. Child class needs to implement ``build`` and ``load``.

    Args:
        dataset_name (str): Name of the dataset passed from child.
    """

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name):
        self._dataset_name = dataset_name

    def build_dataset(self, config, dataset_type="train", *args, **kwargs):
        """
        Similar to load function, used by MMF to build a dataset for first
        time when it is not available. This internally calls 'build' function.
        Override that function in your child class.

        Args:
            config (DictConfig): Configuration of this dataset loaded from
                                 config.
            dataset_type (str): Type of dataset, train|val|test

        .. warning::

            DO NOT OVERRIDE in child class. Instead override ``build``.
        """
        # Only build in main process, so none of the others have to build
        if is_master():
            self.build(config, dataset_type, *args, **kwargs)
        synchronize()

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
