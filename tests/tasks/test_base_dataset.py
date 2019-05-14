# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
import os

from pythia.common.registry import registry
from pythia.tasks.base_dataset import BaseDataset
from pythia.utils.configuration import Configuration


class TestBaseDataset(unittest.TestCase):
    def test_init_processors(self):
        path = os.path.join(
            os.path.abspath(__file__),
            "../../../pythia/common/defaults/configs/tasks/vqa/vqa2.yml"
        )

        configuration = Configuration(os.path.abspath(path))
        self._fix_configuration(configuration)
        configuration.freeze()

        base_dataset = BaseDataset(
            "vqa",
            "vqa2",
            configuration.get_config()["task_attributes"]["vqa"]["dataset_attributes"][
                "vqa2"
            ],
        )
        expected_processors = [
            "answer_processor",
            "ocr_token_processor",
            "bbox_processor",
        ]

        # Check no processors are initialized before init_processors call
        self.assertFalse(any(hasattr(base_dataset, key)
                             for key in expected_processors))

        for processor in expected_processors:
            self.assertIsNone(registry.get("{}_{}".format("vqa", processor)))

        # Check processors are initialized after init_processors
        base_dataset.init_processors()
        self.assertTrue(all(hasattr(base_dataset, key)
                            for key in expected_processors))
        for processor in expected_processors:
            self.assertIsNotNone(registry.get("{}_{}".format("vqa", processor)))

    def _fix_configuration(self, configuration):
        vqa_config = configuration.config['task_attributes']['vqa']
        vqa2_config = vqa_config['dataset_attributes']['vqa2']
        processors = vqa2_config['processors']
        processors.pop('text_processor')
        processors.pop('context_processor')
