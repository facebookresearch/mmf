# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.common.report import Report
from mmf.common.sample import SampleList
from mmf.datasets.processors.prediction_processors import ArgMaxPredictionProcessor


class TestDatasetProcessors(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_argmax_prediction_processor(self):
        processor = ArgMaxPredictionProcessor(config={})
        batch = SampleList({"id": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)})
        model_output = {"scores": torch.rand(5, 4)}
        report = Report(batch, model_output)

        predictions = processor(report)

        expected_answers = [1, 1, 2, 1, 3]
        expected = []
        for idx, answer in enumerate(expected_answers):
            expected.append({"id": idx + 1, "answer": answer})

        self.assertEqual(predictions, expected)
