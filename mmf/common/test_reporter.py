# Copyright (c) Facebook, Inc. and its affiliates.
import csv
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import List

import pytorch_lightning as pl
from mmf.common.registry import registry
from mmf.common.sample import convert_batch_to_sample_list
from mmf.utils.configuration import get_mmf_env
from mmf.utils.distributed import gather_tensor, is_main
from mmf.utils.file_io import PathManager
from mmf.utils.general import ckpt_name_from_core_args, foldername_from_config_override
from mmf.utils.logger import log_class_usage
from mmf.utils.timer import Timer
from omegaconf import OmegaConf
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_FIELDS = [
    "id",
    "question_id",
    "image_id",
    "context_tokens",
    "captions",
    "scores",
]


@registry.register_test_reporter("file")
@registry.register_test_reporter("default")
class TestReporter(Dataset):
    @dataclass
    class Config:
        # A set of fields to be *considered* for exporting by the reporter
        # Note that `format_for_prediction` is what ultimtly detemrimes the
        # exported fields
        candidate_fields: List[str] = field(
            default_factory=lambda: DEFAULT_CANDIDATE_FIELDS
        )
        # csv or json
        predict_file_format: str = "json"

    def __init__(
        self,
        datamodules: List[pl.LightningDataModule],
        config: Config = None,
        dataset_type: str = "train",
    ):
        self.test_reporter_config = OmegaConf.merge(
            OmegaConf.structured(self.Config), config
        )
        self.datamodules = datamodules
        self.dataset_type = dataset_type
        self.config = registry.get("config")
        self.report = []
        self.timer = Timer()
        self.training_config = self.config.training
        self.num_workers = self.training_config.num_workers
        self.batch_size = self.training_config.batch_size
        self.report_folder_arg = get_mmf_env(key="report_dir")
        self.experiment_name = self.training_config.experiment_name

        self.current_datamodule_idx = -1
        self.dataset_names = list(self.datamodules.keys())
        self.current_datamodule = self.datamodules[
            self.dataset_names[self.current_datamodule_idx]
        ]
        self.current_dataloader = None

        self.save_dir = get_mmf_env(key="save_dir")
        self.report_folder = ckpt_name_from_core_args(self.config)
        self.report_folder += foldername_from_config_override(self.config)

        self.report_folder = os.path.join(self.save_dir, self.report_folder)
        self.report_folder = os.path.join(self.report_folder, "reports")

        if self.report_folder_arg:
            self.report_folder = self.report_folder_arg

        self.candidate_fields = self.test_reporter_config.candidate_fields

        PathManager.mkdirs(self.report_folder)

        log_class_usage("TestReporter", self.__class__)

    @property
    def current_dataset(self):
        self._check_current_dataloader()
        return self.current_dataloader.dataset

    def next_dataset(self, flush_report=True):
        if self.current_datamodule_idx >= 0:
            if flush_report:
                self.flush_report()
            else:
                self.report = []

        self.current_datamodule_idx += 1

        if self.current_datamodule_idx == len(self.datamodules):
            return False
        else:
            self.current_datamodule = self.datamodules[
                self.dataset_names[self.current_datamodule_idx]
            ]
            logger.info(
                f"Predicting for {self.dataset_names[self.current_datamodule_idx]}"
            )
            return True

    def flush_report(self):
        if not is_main():
            # Empty report in all processes to avoid any leaks
            self.report = []
            return

        name = self.current_datamodule.dataset_name
        time_format = "%Y-%m-%dT%H:%M:%S"
        time = self.timer.get_time_hhmmss(None, format=time_format)

        filename = name + "_"

        if len(self.experiment_name) > 0:
            filename += self.experiment_name + "_"

        filename += self.dataset_type + "_"
        filename += time

        use_csv_writer = (
            self.config.evaluation.predict_file_format == "csv"
            or self.test_reporter_config.predict_file_format == "csv"
        )

        if use_csv_writer:
            filepath = os.path.join(self.report_folder, filename + ".csv")
            self.csv_dump(filepath)
        else:
            filepath = os.path.join(self.report_folder, filename + ".json")
            self.json_dump(filepath)

        logger.info(f"Wrote predictions for {name} to {os.path.abspath(filepath)}")
        self.report = []

    def postprocess_dataset_report(self):
        self._check_current_dataloader()
        if hasattr(self.current_dataset, "on_prediction_end"):
            self.report = self.current_dataset.on_prediction_end(self.report)

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
        self.current_dataloader = getattr(
            self.current_datamodule, f"{self.dataset_type}_dataloader"
        )()
        # Make sure to assign dataset to dataloader object as
        # required by MMF
        if not hasattr(self.current_dataloader, "dataset"):
            self.current_dataloader.dataset = getattr(
                self.current_datamodule, f"{self.dataset_type}_dataset"
            )
        return self.current_dataloader

    def prepare_batch(self, batch):
        self._check_current_dataloader()
        if hasattr(self.current_dataset, "prepare_batch"):
            batch = self.current_dataset.prepare_batch(batch)

        batch = convert_batch_to_sample_list(batch)
        batch.dataset_name = self.current_dataset.dataset_name
        batch.dataset_type = self.dataset_type
        return batch

    def __len__(self):
        self._check_current_dataloader()
        return len(self.current_dataloader)

    def _check_current_dataloader(self):
        assert self.current_dataloader is not None, (
            "Please call `get_dataloader` before accessing any "
            + "'current_dataloader' based function"
        )

    def add_to_report(self, report, model, *args, **kwargs):
        if "execute_on_master_only" in kwargs:
            warnings.warn(
                "'execute_on_master_only keyword is deprecated and isn't used anymore",
                DeprecationWarning,
            )
        self._check_current_dataloader()
        for key in self.candidate_fields:
            report = self.reshape_and_gather(report, key)

        results = []

        if hasattr(self.current_dataset, "format_for_prediction"):
            results = self.current_dataset.format_for_prediction(report)

        if hasattr(model, "format_for_prediction"):
            results = model.format_for_prediction(results, report)
        elif hasattr(model.module, "format_for_prediction"):
            results = model.module.format_for_prediction(results, report)

        self.report = self.report + results

    def reshape_and_gather(self, report, key):
        if key in report:
            num_dims = report[key].dim()
            if num_dims == 1:
                report[key] = gather_tensor(report[key]).view(-1)
            elif num_dims >= 2:
                # Collect dims other than batch
                other_dims = report[key].size()[1:]
                report[key] = gather_tensor(report[key]).view(-1, *other_dims)

        return report
