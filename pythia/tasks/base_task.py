# Copyright (c) Facebook, Inc. and its affiliates.
"""
Tasks come above datasets in hierarchy level. In case you want to
implement a new task, you need to inherit ``BaseTask`` class. You need
to implement ``_get_available_datasets`` and ``_preprocess_item`` functions
to complete the implementation. You can check the source to see if you need
to override any other methods like ``prepare_batch``.

Check example of ``VQATask`` here_.

Example::

    from pythia.tasks.base_task import BaseTask
    from pythia.common.registry import registry


    @registry.register_task("my")
    class MyTask(BaseTask):
        def __init__(self):
            super().__init__("my")

        def _get_available_datasets(self):
            return ["my"]

        def _preprocess_item(self):
            item.text = None
            return item

.. _here: https://github.com/facebookresearch/pythia/blob/v0.3/pythia/tasks/vqa/vqa_task.py
"""

import sys

import numpy as np
from torch.utils.data import Dataset

from pythia.common.registry import registry
from pythia.utils.distributed_utils import synchronize, is_main_process


class BaseTask(Dataset):
    """
    BaseTask that task classes need to inherit in order to create a new task.

    Users must implement ``_get_available_datasets`` and ``_preprocess_item``
    in order to complete implementation.

    Args:
        task_name (str): Name of the task with which it will be registered
    """

    def __init__(self, task_name):
        super(BaseTask, self).__init__()
        self.task_name = task_name
        self.writer = registry.get("writer")

    def _process_datasets(self):
        if "datasets" not in self.opts:
            self.writer.write(
                "No datasets attribute present for task: %s."
                " Defaulting to all" % (self.task_name),
                "warning",
            )
            datasets = "all"
        else:
            datasets = self.opts["datasets"]

        if datasets is None or datasets == "all":
            datasets = self._get_available_datasets()

        if type(datasets) == str:
            datasets = list(map(lambda x: x.strip(), datasets.split(",")))

        if len(datasets) == 0 and datasets[0] == "all":
            datasets = self._get_available_datasets()

        self.given_datasets = datasets

    def load(self, **opts):
        self.opts = opts
        self._process_datasets()

        self.datasets = []
        self.builders = []
        available_datasets = self._get_available_datasets()

        self.total_length = 0
        self.per_dataset_lengths = []
        self.num_datasets = 0

        for dataset in self.given_datasets:
            if dataset in available_datasets:
                builder_class = registry.get_builder_class(dataset)

                if builder_class is None:
                    print("No builder class found for %s." % dataset)
                    continue
                builder_instance = builder_class()

                if dataset in self.opts["dataset_attributes"]:
                    attributes = self.opts["dataset_attributes"][dataset]
                else:
                    self.writer.write(
                        "Dataset %s is missing from "
                        "dataset_attributes in config." % dataset,
                        "error",
                    )
                    sys.exit(1)

                dataset_type = self.opts.get("dataset_type", "train")
                builder_instance.build(dataset_type, attributes)
                dataset_instance = builder_instance.load(dataset_type, attributes)

                if dataset_instance is None:
                    continue

                self.builders.append(builder_instance)
                self.datasets.append(dataset_instance)
                self.per_dataset_lengths.append(len(dataset_instance))
                self.total_length += len(dataset_instance)
            else:
                print(
                    "Dataset %s is not a valid dataset for task %s. Skipping"
                    % (dataset, self.task_name)
                )

        self.num_datasets = len(self.datasets)
        self.dataset_probablities = [1 for _ in range(self.num_datasets)]
        sampling = self.opts.get("dataset_size_proportional_sampling", None)

        if sampling is True:
            self.dataset_probablities = self.per_dataset_lengths[:]
            self.dataset_probablities = [
                prob / self.total_length for prob in self.dataset_probablities
            ]

        self.change_dataset()

    def _get_available_datasets(self):
        """Set available datasets for this task here.
        Override in your child task class
        Temporary solution, later we will use decorators to easily register
        datasets with a task

        Returns:
            List - List of available datasets for this particular task
        """
        return []

    def get_datasets(self):
        return self.datasets

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        idx = idx % self.per_dataset_lengths[self.dataset_choice]

        item = self.chosen_dataset[idx]

        return self._preprocess_item(item)

    def change_dataset(self):
        self.dataset_choice = np.random.choice(
            self.num_datasets, 1, p=self.dataset_probablities
        )[0]
        self.chosen_dataset = self.datasets[self.dataset_choice]

    def verbose_dump(self, *args, **kwargs):
        self.chosen_dataset.verbose_dump(*args, **kwargs)

    def prepare_batch(self, batch):
        return self.chosen_dataset.prepare_batch(batch)

    def _preprocess_item(self, item):
        """Preprocess an item to be returned from __getitem__.
        Override in your child task class, so you have control on what you are
        returning

        Args:
            item (Sample): Sample returned by a particular dataset

        Returns:
            Sample: Preprocessed item
        """
        raise NotImplementedError(
            "This task doesn't implement preprocess_item" " method"
        )

    def update_registry_for_model(self, config):
        """
        Use this if there is some specific configuration required by model
        which must be inferred at runtime.
        """
        for builder in self.builders:
            builder.update_registry_for_model(config)

    def init_args(self, parser):
        parser.add_argument_group("General Task Arguments")
        parser.add_argument(
            "-dsp",
            "--dataset_size_proportional_sampling",
            type=bool,
            default=0,
            help="Pass if you want to sample from"
            " dataset according to its size. Default: Equal "
            " weighted sampling",
        )

        # TODO: Figure out later if we want to init args from datasets
        # self._init_args(parser)

    def _init_args(self, parser):
        """Override this function to add extra parameters to
        parser in your child task class.

        Parameters
        ----------
        parser : ArgumentParser
            Original parser object passed from the higher level classes like
            trainer

        Returns
        -------
        type
            Description of returned object.

        """
        for builder in self.builders:
            builder.init_args(parser)

    def clean_config(self, config):
        """
        Override this in case you want to clean the config you updated earlier
        in update_registry_for_model
        """
        return config
