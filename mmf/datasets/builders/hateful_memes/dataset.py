# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import os

import numpy as np
import omegaconf
import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.general import get_mmf_root
from mmf.utils.visualize import visualize_images
from PIL import Image
from torchvision import transforms


class HatefulMemesFeaturesDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="hateful_memes", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        assert (
            self._use_features
        ), "config's 'use_images' must be true to use image dataset"

    def preprocess_sample_info(self, sample_info):
        image_path = sample_info["img"]
        # img/02345.png -> 02345
        feature_path = image_path.split("/")[-1].split(".")[0]
        # Add feature_path key for feature_database access
        sample_info["feature_path"] = f"{feature_path}.npy"
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
        if hasattr(self, "transformer_bbox_processor"):
            features["image_info_0"] = self.transformer_bbox_processor(
                features["image_info_0"]
            )
        current_sample.update(features)

        if "label" in sample_info:
            current_sample.targets = torch.tensor(
                sample_info["label"], dtype=torch.long
            )

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

        if "label" in sample_info:
            current_sample.targets = torch.tensor(
                sample_info["label"], dtype=torch.long
            )

        return current_sample

    def format_for_prediction(self, report):
        return generate_prediction(report)

    def visualize(self, num_samples=1, use_transforms=False, *args, **kwargs):
        image_paths = []
        random_samples = np.random.randint(0, len(self), size=num_samples)

        for idx in random_samples:
            image_paths.append(self.annotation_db[idx]["img"])

        images = self.image_db.from_path(image_paths, use_transforms=use_transforms)
        visualize_images(images["images"], *args, **kwargs)


def generate_prediction(report):
    scores = torch.nn.functional.softmax(report.scores, dim=1)
    _, labels = torch.max(scores, 1)
    # Probability that the meme is hateful, (1)
    probabilities = scores[:, 1]

    predictions = []

    for idx, image_id in enumerate(report.id):
        proba = probabilities[idx].item()
        label = labels[idx].item()
        predictions.append({"id": image_id.item(), "proba": proba, "label": label})
    return predictions
