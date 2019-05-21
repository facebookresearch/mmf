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


class COCODataset(BaseDataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__("coco", dataset_type, config)
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
        self._should_fast_read = self.config.fast_read

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

        if self._dataset_type != "test":
            text_processor_argument = {"tokens": sample_info["caption_tokens"]}
            processed_caption = self.text_processor(text_processor_argument)
            current_sample.text = processed_caption["text"]
            current_sample.caption_id = torch.tensor(
                sample_info["caption_id"], dtype=torch.int
            )
        current_sample.text_len = torch.tensor(
            len(sample_info["caption_tokens"]), dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["image_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["image_id"]

        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)

        # Add reference captions to sample
        current_sample = self.add_reference_caption(sample_info, current_sample)

        return current_sample

    def add_reference_caption(self, sample_info, sample):
        reference_list = []
        for reference in sample_info["reference_tokens"]:
            text_processor_argument = {"tokens": reference}
            processed_reference = self.text_processor(text_processor_argument)
            reference_list.append(processed_reference["text"])

        # Restrict to 5 reference captions
        sample.answers = torch.stack(reference_list)[:5]

        return sample

    def format_for_evalai(self, report):
        vocab = self.text_processor.vocab
        captions = report.captions.tolist()
        predictions = []
        for idx, image_id in enumerate(report.image_id):
            for idx_, el in enumerate(captions[idx]):
                if el == vocab.EOS_INDEX:
                    captions[idx] = captions[idx][:idx_]
                    break
            answer = [
                vocab.get_itos()[w]
                for w in captions[idx]
                if w not in {vocab.SOS_INDEX, vocab.EOS_INDEX, vocab.PAD_INDEX}
            ]
            answer = " ".join(answer)
            predictions.append({"image_id": image_id.item(), "caption": answer})

        return predictions
