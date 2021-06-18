# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import os
import unittest
from unittest.mock import MagicMock

import torch
from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from mmf.utils.general import get_current_device
from omegaconf import OmegaConf


class SimpleMMFDataset(BaseDataset):
    def __init__(
        self, dataset_name, config, dataset_type, num_examples, *args, **kwargs
    ):
        self.num_examples = num_examples
        self.features = [float(x) for x in range(self.num_examples)]
        self.annotations = [float(x) for x in range(self.num_examples)]
        self._device = get_current_device()
        self._dataset_name = dataset_name

    def __getitem__(self, idx):
        return {
            "feature": torch.tensor(self.features[idx]),
            "annotation": torch.tensor(self.annotations[idx]),
            "test": torch.tensor(self.features[idx]),
        }

    def __len__(self):
        return self.num_examples


class TestMMFDatasetBuilder(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {
                "use_features": True,
                "use_images": False,
                "split_train": {"val": 0.2, "test": 0.1, "seed": 42},
                "annotations": {"train": "not_a_real_annotations_dataset"},
                "features": {"train": "not_a_real_features_dataset"},
            }
        )
        self.train = self._create_dataset("train")
        self.val = self._create_dataset("val")
        self.test = self._create_dataset("test")

    def test_train_split_len(self):
        self.assertEqual(len(self.train), 70)
        self.assertEqual(len(self.val), 20)
        self.assertEqual(len(self.test), 10)

    def test_train_split_non_overlap(self):
        total = (
            self._samples_set(self.train)
            | self._samples_set(self.val)
            | self._samples_set(self.test)
        )
        self.assertSetEqual(total, {x for x in range(100)})

    def test_train_split_alignment(self):
        self._test_alignment_in_dataset(self.train)
        self._test_alignment_in_dataset(self.val)
        self._test_alignment_in_dataset(self.test)

    def _create_dataset(self, dataset_type):
        dataset_builder = MMFDatasetBuilder(
            "vqa", functools.partial(SimpleMMFDataset, num_examples=100)
        )
        return dataset_builder.load(self.config, dataset_type)

    def _samples_set(self, dataset):
        return set(dataset.features)

    def _test_alignment_in_dataset(self, dataset):
        for feature, annotation in zip(dataset.features, dataset.annotations):
            self.assertEqual(feature, annotation)
