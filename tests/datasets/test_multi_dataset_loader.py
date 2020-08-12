# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
from collections import Counter

import numpy as np
import torch
from mmf.datasets.multi_dataset_loader import MultiDatasetLoader
from torch.utils.data import DataLoader

from ..test_utils import NumbersDataset


class TestMultiDatasetLoader(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        np.random.seed(1234)
        self.multi_dataset = MultiDatasetLoader()
        self.multi_dataset._num_datasets = 3
        self.multi_dataset.current_index = 0
        numbers_dataset_a = NumbersDataset(4, "a")
        numbers_dataset_b = NumbersDataset(40, "b")
        numbers_dataset_c = NumbersDataset(4000, "c")
        self.multi_dataset._datasets = [
            numbers_dataset_a,
            numbers_dataset_b,
            numbers_dataset_c,
        ]
        self.multi_dataset._loaders = [
            self._get_dataloader(numbers_dataset_a),
            self._get_dataloader(numbers_dataset_b),
            self._get_dataloader(numbers_dataset_c),
        ]
        self.multi_dataset.current_loader = self.multi_dataset.loaders[0]
        self.multi_dataset.config = {
            "training": {"dataset_size_proportional_sampling": True, "max_epochs": None}
        }
        self.multi_dataset._per_dataset_lengths = [4, 40, 4000]
        self.multi_dataset._total_length = sum(self.multi_dataset._per_dataset_lengths)

    def _get_dataloader(self, dataset):
        return DataLoader(dataset=dataset, batch_size=4, num_workers=0)

    def test_proportional_sampling(self):
        self.multi_dataset._infer_dataset_probabilities()

        count = 0
        count_c = 0
        for batch in self.multi_dataset:
            batch = self.multi_dataset.prepare_batch(batch)
            if "c" in batch:
                count_c += 1
            count += 1
            if count == 100:
                break

        # Expect more than 95 c's at least as the len for c is very high
        self.assertTrue(count_c >= 98)

        count = 0
        count_epoch = 0
        counter = Counter()
        for _ in range(1):
            for batch in self.multi_dataset:
                batch = self.multi_dataset.prepare_batch(batch)
                counter[list(batch.keys())[0]] += 1
                count += 1
            count_epoch += 1
        # Expect epoch to be completed
        self.assertEqual(count_epoch, 1)
        # Expect each dataset to be full iterated
        self.assertEqual(count, self.multi_dataset._total_length // 4)
        self.assertEqual(counter, Counter({"a": 1, "b": 10, "c": 1000}))

    def test_equal_sampling(self):
        self.multi_dataset.config["training"][
            "dataset_size_proportional_sampling"
        ] = False
        self.multi_dataset._infer_dataset_probabilities()

        count = 0
        count_c = 0
        for batch in self.multi_dataset:
            batch = self.multi_dataset.prepare_batch(batch)
            if "c" in batch:
                count_c += 1
            count += 1
            if count == 100:
                break

        self.assertTrue(count_c <= 34)

        # Epoch will never finish for this case, so test upto proportional sampling's
        # epoch length + some extra
        for batch in self.multi_dataset:
            batch = self.multi_dataset.prepare_batch(batch)
            count += 1
            if count > self.multi_dataset._total_length // 4 + 100:
                break

        # The test should reach at this stage and should not be finished at
        # epoch length
        self.assertTrue(count > self.multi_dataset._total_length // 4 + 100)
