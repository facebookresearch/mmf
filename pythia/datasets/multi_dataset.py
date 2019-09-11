# Copyright (c) Facebook, Inc. and its affiliates.
"""
MultiDataset class is used by DatasetLoader class to load multiple datasets and more granular
"""

import sys

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pythia.common.batch_collator import BatchCollator
from pythia.common.registry import registry
from pythia.datasets.samplers import DistributedSampler
from pythia.utils.distributed_utils import synchronize, is_main_process, broadcast_scalar
from pythia.utils.general import get_batch_size


class MultiDataset:
    """
    MultiDataset class that is used for training on multiple datasets together.
    """
    def __init__(self, dataset_type="train"):
        self._dataset_type = dataset_type
        self.writer = registry.get("writer")
        self._is_main_process = is_main_process()
        self._global_config = registry.get("config")


    def _process_datasets(self):
        if "datasets" not in self.opts:
            self.writer.write(
                "No datasets attribute present. Setting default to vqa2."
                "warning",
            )
            datasets = "vqa2"
        else:
            datasets = self.opts["datasets"]

        if type(datasets) == str:
            datasets = list(map(lambda x: x.strip(), datasets.split(",")))

        self._given_datasets = datasets

    def load(self, **opts):
        self.opts = opts
        self._process_datasets()

        self._datasets = []
        self._builders = []
        self._loaders = []
        self._samplers = []
        self._iterators = []

        self._total_length = 0
        self._per_dataset_lengths = []
        self._num_datasets = 0
        self._finished_iterators = {}
        self._used_once = {}

        for dataset in self._given_datasets:
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


            builder_instance.build(self._dataset_type, attributes)
            dataset_instance = builder_instance.load(self._dataset_type, attributes)

            if dataset_instance is None:
                continue

            loader_instance, sampler_instance = self.build_dataloader(
                dataset_instance, self.opts
            )

            self._builders.append(builder_instance)
            self._datasets.append(dataset_instance)
            self._loaders.append(loader_instance)
            self._samplers.append(sampler_instance)

            self._per_dataset_lengths.append(len(dataset_instance))
            self._total_length += len(dataset_instance)

        self._num_datasets = len(self._datasets)
        self._dataset_probablities = [
            1 / self._num_datasets for _ in range(self._num_datasets)
        ]

        training_parameters = self._global_config.training_parameters
        self._proportional_sampling = training_parameters.dataset_size_proportional_sampling

        if self._dataset_type != "train":
            # If it is val or test, it needs to be all datasets need to be fully iterated
            # as metrics will be calculated in eval mode over complete datasets
            self._proportional_sampling = True

        if self._proportional_sampling is True:
            self._dataset_probablities = self._per_dataset_lengths[:]
            self._dataset_probablities = [
                prob / self._total_length for prob in self._dataset_probablities
            ]

        self._loader_index = 0
        self._chosen_dataset = self._datasets[self._loader_index]
        self._chosen_loader = self._loaders[self._loader_index]

    @property
    def dataset_type(self):
        return self._dataset_type

    @property
    def num_datasets(self):
        return self._num_datasets

    def get_datasets(self):
        return self._datasets

    @property
    def first_loader(self):
        return self._loaders[0]

    def __len__(self):
        # Since, this is iterator, we need to return total length == number of batches
        return self._total_length // get_batch_size()

    def __iter__(self):
        if self._num_datasets == 1:
            return iter(self._loaders[0])

        self._iterators = []
        self._finished_iterators = {}
        self._used_once = {}

        for loader in self._loaders:
            self._iterators.append(iter(loader))

        self._chosen_iterator = self._iterators[self._loader_index]

        return self

    def __next__(self):
        try:
            next_batch = next(self._chosen_iterator)
        except StopIteration:
            if (
                self._proportional_sampling is True or
                len(self._used_once) != self._num_datasets
            ):
                self._finished_iterators[self._loader_index] = 1

                if len(self._finished_iterators) == self._num_datasets:
                    raise
                else:
                    self.change_dataloader()
                next_batch = next(self._chosen_iterator)
            else:
                raise

        self._used_once[self._loader_index] = 1
        return next_batch

    def change_dataloader(self):
        if self._num_datasets <= 1:
            return
        choice = 0

        if self._is_main_process:
            choice = np.random.choice(
                self._num_datasets, 1, p=self._dataset_probablities
            )[0]

            while choice in self._finished_iterators:
                choice = np.random.choice(
                    self._num_datasets, 1, p=self._dataset_probablities
                )[0]

        choice = broadcast_scalar(choice, 0, device=registry.get("current_device"))
        self._loader_index = choice
        self._chosen_dataset = self._datasets[self._loader_index]
        self._chosen_loader = self._loaders[self._loader_index]
        self._chosen_iterator = self._iterators[self._loader_index]

    def verbose_dump(self, *args, **kwargs):
        self._chosen_dataset.verbose_dump(*args, **kwargs)

    def prepare_batch(self, batch):
        batch = self._chosen_dataset.prepare_batch(batch)
        self.change_dataloader()
        return batch

    def update_registry_for_model(self, config):
        """
        Use this if there is some specific configuration required by model
        which must be inferred at runtime.
        """
        for builder in self._builders:
            builder.update_registry_for_model(config)

    def init_args(self, parser):
        parser.add_argument_group("General MultiDataset Arguments")
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
        for builder in self._builders:
            builder.init_args(parser)

    def clean_config(self, config):
        """
        Override this in case you want to clean the config you updated earlier
        in update_registry_for_model
        """
        return config

    def build_dataloader(self, dataset, opts):
        training_parameters = self._global_config.training_parameters
        num_workers = training_parameters.num_workers
        pin_memory = training_parameters.pin_memory

        other_args = {}

        self._add_extra_args_for_dataloader(dataset, opts, other_args)

        loader = DataLoader(
            dataset=dataset,
            pin_memory=pin_memory,
            collate_fn=BatchCollator(),
            num_workers=num_workers,
            **other_args
        )
        loader.dataset_type = self._dataset_type

        return loader, other_args.get("sampler", None)

    def _add_extra_args_for_dataloader(self, dataset, opts, other_args={}):
        training_parameters = self._global_config.training_parameters
        dataset_type = self._dataset_type

        other_args["shuffle"] = False
        if dataset_type != "test":
            other_args["shuffle"] = True

        if (
            training_parameters.local_rank is not None
            and training_parameters.distributed
        ):
            other_args["sampler"] = DistributedSampler(dataset, shuffle=other_args["shuffle"])
            # Shuffle is mutually exclusive with sampler, let DistributedSampler take care of
            # shuffle and pop from main args
            other_args.pop("shuffle")

        other_args["batch_size"] = get_batch_size()

        return other_args

    def seed_sampler(self, epoch):
        training_parameters = self._global_config.training_parameters

        if (
            training_parameters.local_rank is not None
            and training_parameters.distributed
        ):
            for sampler in self._samplers:
                assert hasattr(sampler, "set_epoch"), "Can't seed without `set_epoch` method"
                sampler.set_epoch(epoch)

