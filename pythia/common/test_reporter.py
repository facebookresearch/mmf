# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from pythia.common.batch_collator import BatchCollator
from pythia.common.registry import registry
from pythia.utils.distributed_utils import (gather_tensor, get_world_size,
                                            is_main_process)
from pythia.utils.general import (ckpt_name_from_core_args,
                                  foldername_from_config_override)
from pythia.utils.timer import Timer


class TestReporter(Dataset):
    def __init__(self, multi_task_instance):
        self.test_task = multi_task_instance
        self.task_type = multi_task_instance.dataset_type
        self.config = registry.get("config")
        self.writer = registry.get("writer")
        self.report = []
        self.timer = Timer()
        self.training_parameters = self.config["training_parameters"]
        self.num_workers = self.training_parameters["num_workers"]
        self.batch_size = self.training_parameters["batch_size"]
        self.report_folder_arg = self.config.get("report_folder", None)
        self.experiment_name = self.training_parameters.get("experiment_name", "")

        self.datasets = []

        for task in self.test_task.get_tasks():
            for dataset in task.get_datasets():
                self.datasets.append(dataset)

        self.current_dataset_idx = -1
        self.current_dataset = self.datasets[self.current_dataset_idx]

        self.save_dir = self.config.get("save_dir", "./save")
        self.report_folder = ckpt_name_from_core_args(self.config)
        self.report_folder += foldername_from_config_override(self.config)

        self.report_folder = os.path.join(self.save_dir, self.report_folder)
        self.report_folder = os.path.join(self.report_folder, "reports")

        if self.report_folder_arg is not None:
            self.report_folder = self.report_folder_arg

        if not os.path.exists(self.report_folder):
            os.makedirs(self.report_folder)

    def next_dataset(self):
        if self.current_dataset_idx >= 0:
            self.flush_report()

        self.current_dataset_idx += 1

        if self.current_dataset_idx == len(self.datasets):
            return False
        else:
            self.current_dataset = self.datasets[self.current_dataset_idx]
            self.writer.write("Predicting for " + self.current_dataset._name)
            return True

    def flush_report(self):
        if not is_main_process():
            return

        name = self.current_dataset._name
        time_format = "%Y-%m-%dT%H:%M:%S"
        time = self.timer.get_time_hhmmss(None, format=time_format)

        filename = name + "_"

        if len(self.experiment_name) > 0:
            filename += self.experiment_name + "_"

        filename += self.task_type + "_"

        filename += time + ".json"
        filepath = os.path.join(self.report_folder, filename)

        with open(filepath, "w") as f:
            json.dump(self.report, f)

        self.writer.write(
            "Wrote evalai predictions for %s to %s" % (name, os.path.abspath(filepath))
        )
        self.report = []

    def get_dataloader(self):
        other_args = self._add_extra_args_for_dataloader()
        return DataLoader(
            dataset=self.current_dataset,
            collate_fn=BatchCollator(),
            num_workers=self.num_workers,
            pin_memory=self.config.training_parameters.pin_memory,
            **other_args
        )

    def _add_extra_args_for_dataloader(self, other_args={}):
        training_parameters = self.config.training_parameters

        if (
            training_parameters.local_rank is not None
            and training_parameters.distributed
        ):
            other_args["sampler"] = DistributedSampler(self.current_dataset)
        else:
            other_args["shuffle"] = True

        batch_size = training_parameters.batch_size

        world_size = get_world_size()

        if batch_size % world_size != 0:
            raise RuntimeError(
                "Batch size {} must be divisible by number "
                "of GPUs {} used.".format(batch_size, world_size)
            )

        other_args["batch_size"] = batch_size // world_size

        return other_args

    def prepare_batch(self, batch):
        return self.current_dataset.prepare_batch(batch)

    def __len__(self):
        return len(self.current_dataset)

    def __getitem__(self, idx):
        return self.current_dataset[idx]

    def add_to_report(self, report):
        # TODO: Later gather whole report for no opinions
        if self.current_dataset._name == "coco":
            report.captions = gather_tensor(report.captions)
            report.image_id = gather_tensor(report.image_id).view(-1)
        else:
            report.scores = gather_tensor(report.scores).view(-1, report.scores.size(-1))
            report.question_id = gather_tensor(report.question_id).view(-1)

        if not is_main_process():
            return

        results = self.current_dataset.format_for_evalai(report)

        self.report = self.report + results
