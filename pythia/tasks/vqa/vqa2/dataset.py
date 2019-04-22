# Copyright (c) Facebook, Inc. and its affiliates.
import os

import torch
import tqdm

from pythia.common.sample import Sample
from pythia.tasks.base_dataset import BaseDataset
from pythia.tasks.features_dataset import FeaturesDataset
from pythia.tasks.image_database import ImageDatabase
from pythia.utils.distributed_utils import is_main_process
from pythia.utils.general import get_pythia_root


class VQA2Dataset(BaseDataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__("vqa2", dataset_type, config)
        imdb_files = self.config.imdb_files

        if dataset_type not in imdb_files:
            raise ValueError(
                "Dataset type {} is not present in "
                "imdb_files of dataset config".format(dataset_type)
            )

        self.imdb_file = imdb_files[dataset_type][imdb_file_index]
        self.imdb_file = self._get_absolute_path(self.imdb_file)
        self.imdb = ImageDatabase(self.imdb_file)

        self.kwargs = kwargs
        self.image_depth_first = self.config.image_depth_first
        self._should_fast_read = not self.config.slow_read

        self._use_features = False
        if hasattr(self.config, "image_features"):
            self._use_features = True
            self.features_max_len = self.config.features_max_len

            all_image_feature_dirs = self.config.image_features[dataset_type]
            curr_image_features_dir = all_image_feature_dirs[imdb_file_index]
            curr_image_features_dir = curr_image_features_dir.split(",")
            curr_image_features_dir = self._get_absolute_path(curr_image_features_dir)

            self.features_db = FeaturesDataset(
                "coco",
                directories=curr_image_features_dir,
                depth_first=self.image_depth_first,
                max_features=self.features_max_len,
                fast_read=self._should_fast_read,
                imdb=self.imdb,
            )

    def _get_absolute_path(self, paths):
        if isinstance(paths, list):
            return [self._get_absolute_path(path) for path in paths]
        elif isinstance(paths, str):
            if not os.path.isabs(paths):
                pythia_root = get_pythia_root()
                paths = os.path.join(pythia_root, self.config.data_root_dir, paths)
            return paths
        else:
            raise TypeError(
                "Paths passed to dataset should either be " "string or list"
            )

    def __len__(self):
        return len(self.imdb)

    def try_fast_read(self):
        # Don't fast read in case of test set.
        if self._dataset_type == "test":
            return

        if hasattr(self, "_should_fast_read") and self._should_fast_read is True:
            self.writer.write(
                "Starting to fast read {} {} dataset".format(
                    self._name, self._dataset_type
                )
            )
            self.cache = {}
            for idx in tqdm.tqdm(
                range(len(self.imdb)), miniters=100, disable=not is_main_process()
            ):
                self.cache[idx] = self.load_item(idx)

    def get_item(self, idx):
        if self._should_fast_read is True and self._dataset_type != "test":
            return self.cache[idx]
        else:
            return self.load_item(idx)

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        current_sample = Sample()

        text_processor_argument = {"tokens": sample_info["question_tokens"]}

        processed_question = self.text_processor(text_processor_argument)

        current_sample.text = processed_question["text"]
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["image_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["image_id"]

        current_sample.text_len = torch.tensor(
            len(sample_info["question_tokens"]), dtype=torch.int
        )

        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)

        current_sample = self.add_answer_info(sample_info, current_sample)
        return current_sample

    def add_answer_info(self, sample_info, sample):
        if "answers" in sample_info:
            answer_processor_argument = {"answers": sample_info["answers"]}

            processed_answer = self.answer_processor(answer_processor_argument)
            sample.answers = processed_answer["answers"]
            sample.targets = processed_answer["answers_scores"]
        return sample

    def idx_to_answer(self, idx):
        return self.answer_processor.convert_idx_to_answer(idx)

    def format_for_evalai(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []

        for idx, question_id in enumerate(report.question_id):
            answer = self.answer_processor.idx2word(answers[idx])
            predictions.append({"question_id": question_id.item(), "answer": answer})

        return predictions
