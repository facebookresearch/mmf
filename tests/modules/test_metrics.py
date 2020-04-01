# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest

import torch
import yaml

import pythia.modules.metrics as metrics
from pythia.common.registry import registry
from pythia.common.sample import Sample
from pythia.datasets.processors import CaptionProcessor
from pythia.utils.configuration import load_yaml


class TestModuleMetrics(unittest.TestCase):
    def test_caption_bleu4(self):
        path = os.path.join(
            os.path.abspath(__file__),
            "../../../pythia/common/defaults/configs/datasets/captioning/coco.yml",
        )
        config = load_yaml(os.path.abspath(path))
        captioning_config = config.dataset_config.coco
        caption_processor_config = captioning_config.processors.caption_processor
        vocab_path = os.path.join(
            os.path.abspath(__file__), "..", "..", "data", "vocab.txt"
        )
        caption_processor_config.params.vocab.type = "random"
        caption_processor_config.params.vocab.vocab_file = os.path.abspath(vocab_path)
        caption_processor = CaptionProcessor(caption_processor_config.params)
        registry.register("coco_caption_processor", caption_processor)

        caption_bleu4 = metrics.CaptionBleu4Metric()
        expected = Sample()
        predicted = dict()

        # Test complete match
        expected.answers = torch.empty((5, 5, 10))
        expected.answers.fill_(4)
        predicted["scores"] = torch.zeros((5, 10, 19))
        predicted["scores"][:, :, 4] = 1.0

        self.assertEqual(caption_bleu4.calculate(expected, predicted).item(), 1.0)

        # Test partial match
        expected.answers = torch.empty((5, 5, 10))
        expected.answers.fill_(4)
        predicted["scores"] = torch.zeros((5, 10, 19))
        predicted["scores"][:, 0:5, 4] = 1.0
        predicted["scores"][:, 5:, 18] = 1.0

        self.assertAlmostEqual(
            caption_bleu4.calculate(expected, predicted).item(), 0.3928, 4
        )
