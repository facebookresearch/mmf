# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest

from pythia.common.registry import registry
from pythia.datasets.base_dataset import BaseDataset
from pythia.utils.configuration import Configuration

from ..test_utils import dummy_args


class TestBaseDataset(unittest.TestCase):
    def test_init_processors(self):
        path = os.path.join(
            os.path.abspath(__file__),
            "../../../pythia/common/defaults/configs/datasets/vqa/vqa2.yml",
        )
        args = dummy_args()
        args.opts.append("config={}".format(path))
        configuration = Configuration(args)
        answer_processor = (
            configuration.get_config().dataset_config.vqa2.processors.answer_processor
        )
        vocab_path = os.path.join(
            os.path.abspath(__file__), "..", "..", "data", "vocab.txt"
        )
        answer_processor.params.vocab_file = os.path.abspath(vocab_path)
        self._fix_configuration(configuration)
        configuration.freeze()

        base_dataset = BaseDataset(
            "vqa2", "train", configuration.get_config().dataset_config.vqa2
        )
        expected_processors = [
            "answer_processor",
            "ocr_token_processor",
            "bbox_processor",
        ]

        # Check no processors are initialized before init_processors call
        self.assertFalse(any(hasattr(base_dataset, key) for key in expected_processors))

        for processor in expected_processors:
            self.assertIsNone(registry.get("{}_{}".format("vqa2", processor)))

        # Check processors are initialized after init_processors
        base_dataset.init_processors()
        self.assertTrue(all(hasattr(base_dataset, key) for key in expected_processors))
        for processor in expected_processors:
            self.assertIsNotNone(registry.get("{}_{}".format("vqa2", processor)))

    def _fix_configuration(self, configuration):
        vqa2_config = configuration.config.dataset_config.vqa2
        processors = vqa2_config.processors
        processors.pop("text_processor")
        processors.pop("context_processor")
