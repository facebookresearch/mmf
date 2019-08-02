# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import warnings

from pythia.common.registry import registry
from pythia.tasks.base_dataset_builder import BaseDatasetBuilder
from pythia.tasks.concat_dataset import PythiaConcatDataset
from pythia.tasks.vqa.vqa2.dataset import VQA2Dataset


@registry.register_builder("vqa2")
class VQA2Builder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("vqa2")
        self.dataset_class = VQA2Dataset

    def _load(self, dataset_type, config, *args, **kwargs):
        self.config = config

        image_features = config["image_features"]["train"][0].split(",")
        self.num_image_features = len(image_features)

        registry.register("num_image_features", self.num_image_features)

        self.dataset = self.prepare_data_set(dataset_type, config)

        return self.dataset

    def _build(self, dataset_type, config):
        # TODO: Build actually here
        return

    def update_registry_for_model(self, config):
        registry.register(
            self.dataset_name + "_text_vocab_size",
            self.dataset.text_processor.get_vocab_size(),
        )
        registry.register(
            self.dataset_name + "_num_final_outputs",
            self.dataset.answer_processor.get_vocab_size(),
        )

    def init_args(self, parser):
        parser.add_argument_group("VQA2 task specific arguments")
        parser.add_argument(
            "--data_root_dir",
            type=str,
            default="../data",
            help="Root directory for data",
        )
        parser.add_argument(
            "-nfr",
            "--fast_read",
            type=bool,
            default=None,
            help="Disable fast read and load features on fly",
        )

    def set_dataset_class(self, cls):
        self.dataset_class = cls

    def prepare_data_set(self, dataset_type, config):
        if dataset_type not in config.imdb_files:
            warnings.warn(
                "Dataset type {} is not present in "
                "imdb_files of dataset config. Returning None. "
                "This dataset won't be used.".format(dataset_type)
            )
            return None

        imdb_files = config["imdb_files"][dataset_type]

        datasets = []

        for imdb_idx in range(len(imdb_files)):
            cls = self.dataset_class
            dataset = cls(dataset_type, imdb_idx, config)
            datasets.append(dataset)

        dataset = PythiaConcatDataset(datasets)

        return dataset
