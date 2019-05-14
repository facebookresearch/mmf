# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from pythia.common.registry import registry
from pythia.tasks.base_dataset import BaseDataset
from pythia.utils.configuration import Configuration


class TestBaseDataset(unittest.TestCase):
    def test_init_processors(self):
        configuration = Configuration(
            "../../common/defaults/configs/tasks/vqa/vqa2.yml"
        )
        configuration.freeze()

        base_dataset = BaseDataset(
            "vqa",
            "vqa2",
            configuration.get_config()["task_attributes"]["vqa"]["dataset_attributes"][
                "vqa2"
            ],
        )
        expected_processors = [
            "text_processor",
            "answer_processor",
            "context_processor",
            "ocr_token_processor",
            "bbox_processor",
        ]

        # Check no processors are initialized before init_processors call
        self.assertFalse(any(hasattr(base_dataset, key) for key in expected_processors))

        for processor in expected_processors:
            self.assertIsNone(registry.get("{}_{}".format("vqa", processor)))

        # Check processors are initialized after init_processors
        base_dataset.init_processors()
        self.assertTrue(all(hasattr(base_dataset, key) for key in expected_processors))
        for processor in expected_processors:
            self.assertIsNotNone(registry.get("{}_{}".format("vqa", processor)))
