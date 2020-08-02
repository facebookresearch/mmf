# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest

from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports
from mmf.utils.general import get_mmf_root
from tests.test_utils import dummy_args


class TestConfigOverrides(unittest.TestCase):
    def setUp(self):
        setup_imports()

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
            "dataset_config.textvqa.zoo_requirements[0]=\"test\""
        ]
        configuration = Configuration(args)
        configuration.freeze()
        config = configuration.get_config()
        self.assertEqual(config.training.lr_steps[1], 10000)
        self.assertEqual(
            config.dataset_config.textvqa.zoo_requirements[0],
            "test"
        )
