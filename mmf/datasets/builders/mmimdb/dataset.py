import copy
import json

import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset


class MMIMDbFeaturesDataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            "mmimdb", config, dataset_type, imdb_file_index, *args, **kwargs
        )
        assert (
            self._use_features
        ), "config's 'use_features' must be true to use feature dataset"

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
        plot = sample_info["plot"]
        if isinstance(plot, list):
            plot = plot[0]
        processed_sentence = self.text_processor({"text": plot})

        current_sample.text = processed_sentence["text"]
        if "input_ids" in processed_sentence:
            current_sample.update(processed_sentence)

        if self._use_features is True:
            features = self.features_db[idx]
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )
            current_sample.update(features)

        processed = self.answer_processor({"answers": sample_info["genres"]})
        current_sample.answers = processed["answers"]
        current_sample.targets = processed["answers_scores"]

        return current_sample


class MMIMDbImageDataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            "mmimdb", config, dataset_type, imdb_file_index, *args, **kwargs
        )
        assert (
            self._use_images
        ), "config's 'use_images' must be true to use image dataset"

    def init_processors(self):
        super().init_processors()
        # Assign transforms to the image_db
        self.image_db.transform = self.image_processor

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
        plot = sample_info["plot"]
        if isinstance(plot, list):
            plot = plot[0]
        processed_sentence = self.text_processor({"text": plot})

        current_sample.text = processed_sentence["text"]
        if "input_ids" in processed_sentence:
            current_sample.update(processed_sentence)

        if self._use_images is True:
            current_sample.image = self.image_db[idx]["images"][0]

        processed = self.answer_processor({"answers": sample_info["genres"]})
        current_sample.answers = processed["answers"]
        current_sample.targets = processed["answers_scores"]

        return current_sample
