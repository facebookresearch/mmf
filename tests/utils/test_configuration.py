# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest

import mmf.utils.configuration as configuration


class TestUtilsConfiguration(unittest.TestCase):
    def test_get_zoo_config(self):
        # Test direct key
        version, resources = configuration.get_zoo_config("textvqa.ocr_en")
        self.assertIsNotNone(version)
        self.assertIsNotNone(resources)

        # Test default variation
        version, resources = configuration.get_zoo_config("textvqa")
        self.assertIsNotNone(version)
        self.assertIsNotNone(resources)

        # Test non-default variation
        version, resources = configuration.get_zoo_config("textvqa", variation="ocr_en")
        self.assertIsNotNone(version)
        self.assertIsNotNone(resources)

        # Test random key
        version, resources = configuration.get_zoo_config("some_random")
        self.assertIsNone(version)
        self.assertIsNone(resources)

        # Test non-existent variation
        self.assertRaises(
            AssertionError,
            configuration.get_zoo_config,
            "textvqa",
            variation="some_random",
        )

        # Test different zoo_type
        version, resources = configuration.get_zoo_config(
            "visual_bert.pretrained", zoo_type="models"
        )
        self.assertIsNotNone(version)
        self.assertIsNotNone(resources)

        # Test direct config
        version, resources = configuration.get_zoo_config(
            "visual_bert.pretrained",
            zoo_config_path=os.path.join("configs", "zoo", "models.yaml"),
        )
        self.assertIsNotNone(version)
        self.assertIsNotNone(resources)
