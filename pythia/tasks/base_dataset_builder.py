# Copyright (c) Facebook, Inc. and its affiliates.
"""
In Pythia, for adding new datasets, dataset builder for datasets need to be
added. A new dataset builder must inherit ``BaseDatasetBuilder`` class and
implement ``_load`` and ``_build`` functions.

``_build`` is used to build a dataset when it is not available. For e.g.
downloading the ImDBs for a dataset. In future, we plan to add a ``_build``
to add dataset builder to ease setup of Pythia.

``_load`` is used to load a dataset from specific path. ``_load`` needs to return
an instance of subclass of ``pythia.tasks.base_dataset.BaseDataset``.

See complete example for ``VQA2DatasetBuilder`` here_.

Example::

    from torch.utils.data import Dataset

    from pythia.tasks.base_dataset_builder import BaseDatasetBuilder
    from pythia.common.registry import registry

    @registry.register_builder("my")
    class MyBuilder(BaseDatasetBuilder):
        def __init__(self):
            super().__init__("my")

        def _load(self, dataset_type, config, *args, **kwargs):
            ...
            return Dataset()

        def _build(self, dataset_type, config, *args, **kwargs):
            ...

.. _here: https://github.com/facebookresearch/pythia/blob/v0.3/pythia/tasks/vqa/vqa2/builder.py
"""


class BaseDatasetBuilder:
    """Base class for implementing dataset builders. See more information
    on top. Child class needs to implement ``_build`` and ``_load``.

    Args:
        dataset_name (str): Name of the dataset passed from child.
    """
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load(self, dataset_type, config, *args, **kwargs):
        """Main load function use by Pythia. This will internally call ``_load``
        function. Calls ``init_processors`` and ``try_fast_read`` on the
        dataset returned from ``_load``

        Args:
            dataset_type (str): Type of dataset, train|val|test
            config (ConfigNode): Configuration of this dataset loaded from config.

        Returns:
            dataset (BaseDataset): Dataset containing data to be trained on

        .. warning::

            DO NOT OVERRIDE in child class. Instead override ``_load``.
        """
        dataset = self._load(dataset_type, config, *args, **kwargs)
        dataset.init_processors()
        dataset.try_fast_read()
        return dataset

    def _load(self, dataset_type, config, *args, **kwargs):
        """
        This is used to prepare the dataset and load it from a path.
        Override this method in your child dataset builder class.

        Args:
            dataset_type (str): Type of dataset, train|val|test
            config (ConfigNode): Configuration of this dataset loaded from config.

        Returns:
            dataset (BaseDataset): Dataset containing data to be trained on
        """
        raise NotImplementedError(
            "This dataset builder doesn't implement a " "load method"
        )

    def build(self, dataset_type, config, *args, **kwargs):
        """
        Similar to load function, used by Pythia to build a dataset for first
        time when it is not available. This internally calls '_build' function.
        Override that function in your child class.

        Args:
            dataset_type (str): Type of dataset, train|val|test
            config (ConfigNode): Configuration of this dataset loaded from
                                 config.

        .. warning::

            DO NOT OVERRIDE in child class. Instead override ``_build``.
        """
        # TODO: Once we start building we will do some preprocessing for folder
        # structure and other things here
        self._build(dataset_type, config, *args, **kwargs)

    def _build(self, dataset_type, config, *args, **kwargs):
        """
        This is used to build a dataset first time.
        Implement this method in your child dataset builder class.

        Args:
            dataset_type (str): Type of dataset, train|val|test
            config (ConfigNode): Configuration of this dataset loaded from
                                 config.
        """
        raise NotImplementedError(
            "This dataset builder doesn't implement a " "build method"
        )
