# Copyright (c) Facebook, Inc. and its affiliates.
import csv
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from mmf.common.batch_collator import BatchCollator
from mmf.common.registry import registry
from mmf.utils.configuration import get_mmf_env
from mmf.utils.distributed import gather_tensor, is_master
from mmf.utils.file_io import PathManager
from mmf.utils.general import (
    ckpt_name_from_core_args,
    foldername_from_config_override,
    get_batch_size,
)
from mmf.utils.timer import Timer


class TestReporter(Dataset):
    def __init__(self, multi_task_instance):
        self.test_task = multi_task_instance
        self.task_type = multi_task_instance.dataset_type
        self.config = registry.get("config")
        self.writer = registry.get("writer")
        self.report = []
        self.timer = Timer()
        self.training_config = self.config.training
        self.num_workers = self.training_config.num_workers
        self.batch_size = self.training_config.batch_size
        self.report_folder_arg = get_mmf_env(key="report_dir")
        self.experiment_name = self.training_config.experiment_name

        self.datasets = []

        for dataset in self.test_task.get_datasets():
            self.datasets.append(dataset)

        self.current_dataset_idx = -1
        self.current_dataset = self.datasets[self.current_dataset_idx]

        self.save_dir = get_mmf_env(key="save_dir")
        self.report_folder = ckpt_name_from_core_args(self.config)
        self.report_folder += foldername_from_config_override(self.config)

        self.report_folder = os.path.join(self.save_dir, self.report_folder)
        self.report_folder = os.path.join(self.report_folder, "reports")

        if self.report_folder_arg:
            self.report_folder = self.report_folder_arg

        PathManager.mkdirs(self.report_folder)

    def next_dataset(self):
        if self.current_dataset_idx >= 0:
            self.flush_report()

        self.current_dataset_idx += 1

        if self.current_dataset_idx == len(self.datasets):
            return False
        else:
            self.current_dataset = self.datasets[self.current_dataset_idx]
            self.writer.write("Predicting for " + self.current_dataset.dataset_name)
            return True

    def flush_report(self):
        if not is_master():
            return

        name = self.current_dataset.dataset_name
        time_format = "%Y-%m-%dT%H:%M:%S"
        time = self.timer.get_time_hhmmss(None, format=time_format)

        filename = name + "_"

        if len(self.experiment_name) > 0:
            filename += self.experiment_name + "_"

        filename += self.task_type + "_"
        filename += time

        if self.config.evaluation.predict_file_format == "csv":
            filepath = os.path.join(self.report_folder, filename + ".csv")
            self.csv_dump(filepath)
        else:
            filepath = os.path.join(self.report_folder, filename + ".json")
            self.json_dump(filepath)

        self.writer.write(
            "Wrote evalai predictions for {} to {}".format(
                name, os.path.abspath(filepath)
            )
        )
        self.report = []

    def csv_dump(self, filepath):
        with PathManager.open(filepath, "w") as f:
            title = self.report[0].keys()
            cw = csv.DictWriter(f, title, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            cw.writeheader()
            cw.writerows(self.report)

    def json_dump(self, filepath):
        with PathManager.open(filepath, "w") as f:
            json.dump(self.report, f)

    def get_dataloader(self):
        other_args = self._add_extra_args_for_dataloader()
        return DataLoader(
            dataset=self.current_dataset,
            collate_fn=BatchCollator(
                self.current_dataset.dataset_name, self.current_dataset.dataset_type
            ),
            num_workers=self.num_workers,
            pin_memory=self.config.training.pin_memory,
            **other_args
        )

    def _add_extra_args_for_dataloader(self, other_args=None):
        if other_args is None:
            other_args = {}

        if torch.distributed.is_initialized():
            other_args["sampler"] = DistributedSampler(
                self.current_dataset, shuffle=False
            )
        else:
            other_args["shuffle"] = False

        other_args["batch_size"] = get_batch_size()

        return other_args

    def prepare_batch(self, batch):
        return self.current_dataset.prepare_batch(batch)

    def __len__(self):
        return len(self.current_dataset)

    def __getitem__(self, idx):
        return self.current_dataset[idx]

    def add_to_report(self, report, model):
        # TODO: Later gather whole report for no opinions
        if self.current_dataset.dataset_name == "coco":
            report.captions = gather_tensor(report.captions)
            if isinstance(report.image_id, torch.Tensor):
                report.image_id = gather_tensor(report.image_id).view(-1)
        else:
            report.scores = gather_tensor(report.scores).view(
                -1, report.scores.size(-1)
            )
            if "id" in report:
                report.id = gather_tensor(report.id).view(-1)
            if "question_id" in report:
                report.question_id = gather_tensor(report.question_id).view(-1)
            if "image_id" in report:
                if report.image_id.dim() == 2:
                    _, enc_size = report.image_id.size()
                    report.image_id = gather_tensor(report.image_id)
                    report.image_id = report.image_id.view(-1, enc_size)
                else:
                    report.image_id = gather_tensor(report.image_id).view(-1)
            if "context_tokens" in report:
                _, enc_size = report.context_tokens.size()
                report.context_tokens = gather_tensor(report.context_tokens)
                report.context_tokens = report.context_tokens.view(-1, enc_size)

        if not is_master():
            return

        results = self.current_dataset.format_for_prediction(report)
        if hasattr(model, "format_for_prediction"):
            results = model.format_for_prediction(results, report)
        elif hasattr(model.module, "format_for_prediction"):
            results = model.module.format_for_prediction(results, report)

        self.report = self.report + results
