# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest

import yaml

import pythia.modules.metrics as metrics
import torch
from pythia.common.registry import registry
from pythia.common.sample import Sample
from pythia.tasks.processors import CaptionProcessor
from pythia.utils.configuration import ConfigNode


class TestModuleMetrics(unittest.TestCase):
    def test_caption_bleu4(self):
        path = os.path.join(
            os.path.abspath(__file__),
            "../../../pythia/common/defaults/configs/tasks/captioning/coco.yml",
        )
        with open(os.path.abspath(path)) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config = ConfigNode(config)
        captioning_config = config.task_attributes.captioning.dataset_attributes.coco
        caption_processor_config = captioning_config.processors.caption_processor
        vocab_path = os.path.join(os.path.abspath(__file__), "..", "..", "data", "vocab.txt")
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

        self.assertAlmostEqual(
            caption_bleu4.calculate(expected, predicted).item(), 0.3928, 4
        )
