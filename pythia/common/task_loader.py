# Copyright (c) Facebook, Inc. and its affiliates.
import os

import yaml
from torch.utils.data import DataLoader

from pythia.common.batch_collator import BatchCollator
from pythia.common.test_reporter import TestReporter
from pythia.tasks import MultiTask
from pythia.tasks.samplers import DistributedSampler
from pythia.utils.distributed_utils import get_world_size


class TaskLoader:
    def __init__(self, config):
        self.config = config

    def load_task(self):
        self.train_task = MultiTask("train", self.config)
        self.val_task = MultiTask("val", self.config)
        self.test_task = MultiTask("test", self.config)

        self.mapping = {
            "train": self.train_task,
            "val": self.val_task,
            "test": self.test_task,
        }

        self.test_reporter = None
        self.should_not_log = self.config.training_parameters.should_not_log

    @property
    def task_config(self):
        return self._task_config

    @task_config.setter
    def task_config(self, config):
        self._task_config = config

    def get_config(self):
        return self._task_config

    def get_test_reporter(self, dataset_type):
        task = getattr(self, "{}_task".format(dataset_type))
        return TestReporter(task)

    def _load_task_config(self, task_name):
        directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(directory, "..", "tasks", task_name, "config.yml")

        self._task_config = {}

        if not os.path.exists(config_path):
            print("[Warning] No config present for task %s" % task_name)
            return {}

        with open(config_path, "r") as f:
            try:
                self._task_config = yaml.load(f)
            except yaml.YAMLError as err:
                print("[Error] Task %s's config yaml error" % task_name, err)

        return self._task_config

    def make_dataloaders(self):
        training_parameters = self.config.training_parameters
        num_workers = training_parameters.num_workers
        pin_memory = training_parameters.pin_memory

        other_args = {}

        self._add_extra_args_for_dataloader(self.train_task, other_args)
        self.train_loader = DataLoader(
            dataset=self.train_task,
            pin_memory=pin_memory,
            collate_fn=BatchCollator(),
            num_workers=num_workers,
            **other_args
        )

        self.train_loader.dataset_type = "train"

        self._add_extra_args_for_dataloader(self.val_task, other_args)
        self.val_loader = DataLoader(
            dataset=self.val_task,
            pin_memory=pin_memory,
            collate_fn=BatchCollator(),
            num_workers=num_workers,
            **other_args
        )
        self.val_loader.dataset_type = "val"

        self._add_extra_args_for_dataloader(self.test_task, other_args)
        self.test_loader = DataLoader(
            dataset=self.test_task,
            pin_memory=pin_memory,
            collate_fn=BatchCollator(),
            num_workers=num_workers,
            **other_args
        )
        self.test_loader.dataset_type = "test"

        self.use_cuda = "cuda" in self.config.training_parameters.device

    def _add_extra_args_for_dataloader(self, task, other_args={}):
        training_parameters = self.config.training_parameters

        other_args["shuffle"] = False
        if task.dataset_type != "test":
            other_args["shuffle"] = True

        if (
            training_parameters.local_rank is not None
            and training_parameters.distributed
        ):
            other_args["sampler"] = DistributedSampler(task, shuffle=other_args["shuffle"])
            # Shuffle is mutually exclusive with sampler, let DistributedSampler take care of
            # shuffle and pop from main args
            other_args.pop("shuffle")
            setattr(self, "{}_sampler".format(task.dataset_type), other_args["sampler"])

        batch_size = training_parameters.batch_size

        world_size = get_world_size()

        if batch_size % world_size != 0:
            raise RuntimeError(
                "Batch size {} must be divisible by number "
                "of GPUs {} used.".format(batch_size, world_size)
            )

        other_args["batch_size"] = batch_size // world_size

        return other_args

    def update_registry_for_model(self, config):
        self.train_task.update_registry_for_model(config)
        self.val_task.update_registry_for_model(config)
        self.test_task.update_registry_for_model(config)

    def clean_config(self, config):
        self.train_task.clean_config(config)
        self.val_task.clean_config(config)
        self.test_task.clean_config(config)

    def prepare_batch(self, batch, *args, **kwargs):
        return self.mapping[batch.dataset_type].prepare_batch(batch)

    def verbose_dump(self, report, *args, **kwargs):
        if self.config.training_parameters.verbose_dump:
            dataset_type = report.dataset_type
            self.mapping[dataset_type].verbose_dump(report, *args, **kwargs)

    def seed_sampler(self, task_type, seed):
        training_parameters = self.config.training_parameters
        if (
            training_parameters.local_rank is not None
            and training_parameters.distributed
        ):
            sampler = getattr(self, "{}_sampler".format(task_type))
            assert hasattr(sampler, "set_epoch"), "Can't seed without `set_epoch` method"
            sampler.set_epoch(seed)
