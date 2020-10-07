# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest

from mmf.utils.configuration import Configuration, get_zoo_config
from mmf.utils.env import setup_imports
from mmf.utils.general import get_mmf_root
from tests.test_utils import dummy_args


class TestUtilsConfiguration(unittest.TestCase):
    def setUp(self):
        setup_imports()

    def test_get_zoo_config(self):
        # Test direct key
        version, resources = get_zoo_config("textvqa.ocr_en")
        self.assertIsNotNone(version)
        self.assertIsNotNone(resources)

        # Test default variation
        version, resources = get_zoo_config("textvqa")
        self.assertIsNotNone(version)
        self.assertIsNotNone(resources)

        # Test non-default variation
        version, resources = get_zoo_config("textvqa", variation="ocr_en")
        self.assertIsNotNone(version)
        self.assertIsNotNone(resources)

        # Test random key
        version, resources = get_zoo_config("some_random")
        self.assertIsNone(version)
        self.assertIsNone(resources)

        # Test non-existent variation
        self.assertRaises(
            AssertionError, get_zoo_config, "textvqa", variation="some_random"
        )

        # Test different zoo_type
        version, resources = get_zoo_config("visual_bert.pretrained", zoo_type="models")
        self.assertIsNotNone(version)
        self.assertIsNotNone(resources)

        # Test direct config
        version, resources = get_zoo_config(
            "visual_bert.pretrained",
            zoo_config_path=os.path.join("configs", "zoo", "models.yaml"),
        )
        self.assertIsNotNone(version)
        self.assertIsNotNone(resources)

    def test_config_overrides(self):
        config_path = os.path.join(
            get_mmf_root(),
            "..",
            "projects",
            "m4c",
            "configs",
            "textvqa",
            "defaults.yaml",
        )
        config_path = os.path.abspath(config_path)
        args = dummy_args(model="m4c", dataset="textvqa")
        args.opts += [
            f"config={config_path}",
            "training.lr_steps[1]=10000",
            'dataset_config.textvqa.zoo_requirements[0]="test"',
        ]
        configuration = Configuration(args)
        configuration.freeze()
        config = configuration.get_config()
        self.assertEqual(config.training.lr_steps[1], 10000)
        self.assertEqual(config.dataset_config.textvqa.zoo_requirements[0], "test")
