# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from collections import Counter

import numpy as np
import torch
from mmf.datasets import iteration_strategies
from tests.test_utils import NumbersDataset


class TestIterationStrategies(unittest.TestCase):
    NUM_DATALOADERS = 5

    def setUp(self):
        np.random.seed(1234)

    def _build_dataloaders(self):
        dataloaders = {}
        for idx in range(self.NUM_DATALOADERS):
            dataloaders[f"numbers_{idx}"] = torch.utils.data.DataLoader(
                dataset=NumbersDataset((idx + 1) * (10**idx)), num_workers=0
            )
        return dataloaders

    def test_constant_iteration_strategy(self):
        dataloaders = self._build_dataloaders()
        strategy = iteration_strategies.ConstantIterationStrategy.from_params(
            dataloaders=dataloaders
        )

        counter = Counter()
        count = 100
        for _ in range(count):
            counter[strategy()] += 1

        self.assertEqual(counter[0], count)
        for idx in range(1, self.NUM_DATALOADERS):
            self.assertEqual(counter[idx], 0)

        strategy = iteration_strategies.ConstantIterationStrategy.from_params(
            dataloaders=dataloaders, idx=1
        )

        counter = Counter()
        count = 100
        for _ in range(count):
            counter[strategy()] += 1

        self.assertEqual(counter[1], count)
        for idx in range(0, self.NUM_DATALOADERS):
            if idx != 1:
                self.assertEqual(counter[idx], 0)

    def test_round_robin_strategy(self):
        dataloaders = self._build_dataloaders()
        strategy = iteration_strategies.RoundRobinIterationStrategy.from_params(
            dataloaders=dataloaders
        )

        counter = Counter()
        count = 100
        for _ in range(count):
            counter[strategy()] += 1

        for idx in range(0, self.NUM_DATALOADERS):
            self.assertEqual(counter[idx], count // self.NUM_DATALOADERS)

        strategy = iteration_strategies.RoundRobinIterationStrategy.from_params(
            dataloaders=dataloaders, start_idx=2
        )
        counter = Counter()
        count = 100
        for _ in range(count):
            counter[strategy()] += 1

        for idx in range(0, self.NUM_DATALOADERS):
            self.assertEqual(counter[idx], count // self.NUM_DATALOADERS)

    def test_random_strategy(self):
        dataloaders = self._build_dataloaders()
        strategy = iteration_strategies.RandomIterationStrategy.from_params(
            dataloaders=dataloaders
        )

        counter = Counter()
        count = 10000
        for _ in range(count):
            counter[strategy()] += 1

        for idx in range(0, self.NUM_DATALOADERS):
            self.assertTrue(counter[idx] <= 2100)
            self.assertTrue(counter[idx] >= 1900)

    def test_size_proportional_strategy(self):
        dataloaders = self._build_dataloaders()
        strategy = iteration_strategies.SizeProportionalIterationStrategy.from_params(
            dataloaders=dataloaders
        )

        counter = Counter()
        count = 10000
        for _ in range(count):
            counter[strategy()] += 1

        for idx in range(0, self.NUM_DATALOADERS):
            self.assertTrue(counter[idx] <= 10**idx)
            lower_limit = 10 ** (idx - 1)
            if idx == 0:
                lower_limit = 0
            self.assertTrue(counter[idx] >= lower_limit)

    def test_ratios_strategy(self):
        dataloaders = self._build_dataloaders()
        sampling_ratios = {}

        # Constant
        for idx in range(self.NUM_DATALOADERS):
            sampling_ratios[f"numbers_{idx}"] = 0
        sampling_ratios["numbers_0"] = 1
        strategy = iteration_strategies.RatiosIterationStrategy.from_params(
            dataloaders, sampling_ratios=sampling_ratios
        )

        counter = Counter()
        count = 10000
        for _ in range(count):
            counter[strategy()] += 1

        self.assertEqual(counter[0], count)
        for idx in range(1, self.NUM_DATALOADERS):
            self.assertEqual(counter[idx], 0)

        for idx in range(self.NUM_DATALOADERS):
            sampling_ratios[f"numbers_{idx}"] = 1.0 / self.NUM_DATALOADERS

        strategy = iteration_strategies.RatiosIterationStrategy.from_params(
            dataloaders, sampling_ratios=sampling_ratios
        )

        count = 10000
        counter = Counter()
        for _ in range(count):
            counter[strategy()] += 1

        for idx in range(0, self.NUM_DATALOADERS):
            self.assertTrue(counter[idx] <= 2100)
            self.assertTrue(counter[idx] >= 1900)

        lens = sum(len(loader.dataset) for loader in dataloaders.values())
        for idx in range(self.NUM_DATALOADERS):
            sampling_ratios[f"numbers_{idx}"] = (
                len(dataloaders[f"numbers_{idx}"].dataset) / lens
            )

        strategy = iteration_strategies.RatiosIterationStrategy.from_params(
            dataloaders, sampling_ratios=sampling_ratios
        )

        count = 10000
        counter = Counter()
        for _ in range(count):
            counter[strategy()] += 1

        for idx in range(0, self.NUM_DATALOADERS):
            self.assertTrue(counter[idx] <= 10**idx)
            lower_limit = 10 ** (idx - 1)
            if idx == 0:
                lower_limit = 0
            self.assertTrue(counter[idx] >= lower_limit)
