# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import os

import omegaconf
import torch
from PIL import Image
from torchvision import transforms

from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.general import get_mmf_root


class HatefulMemesFeaturesDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="hateful_memes", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        assert (
            self._use_features
        ), "config's 'use_images' must be true to use image dataset"

    def preprocess_sample_info(self, sample_info):
        image_id = sample_info["id"]
        # Add feature_path key for feature_database access
        sample_info["feature_path"] = f"{image_id}.npy"
        return sample_info

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info = self.preprocess_sample_info(sample_info)

        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        if "input_ids" in processed_text:
            current_sample.update(processed_text)

        current_sample.id = torch.tensor(int(sample_info["id"]), dtype=torch.int)

        # Instead of using idx directly here, use sample_info to fetch
        # the features as feature_path has been dynamically added
        features = self.features_db.get(sample_info)
        current_sample.update(features)

        current_sample.targets = torch.tensor(sample_info["label"], dtype=torch.long)
        return current_sample

    def format_for_prediction(self, report):
        return generate_prediction(report)


class HatefulMemesImageDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="hateful_memes", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
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

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        if "input_ids" in processed_text:
            current_sample.update(processed_text)

        current_sample.id = torch.tensor(int(sample_info["id"]), dtype=torch.int)

        # Get the first image from the set of images returned from the image_db
        current_sample.image = self.image_db[idx]["images"][0]
        current_sample.targets = torch.tensor(sample_info["label"], dtype=torch.long)
        return current_sample

    def format_for_prediction(self, report):
        return generate_prediction(report)


def generate_prediction(report):
    probabilities, labels = torch.max(
        torch.nn.functional.softmax(report.scores, dim=1), 1
    )

    predictions = []

    for idx, image_id in enumerate(report.id):
        proba = probabilities[idx].item()
        label = labels[idx].item()
        predictions.append(
            {"id": image_id.item(), "proba": proba, "label": label,}
        )
    return predictions
