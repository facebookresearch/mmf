# Copyright (c) Facebook, Inc. and its affiliates.

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tests.test_utils as test_utils
import torch
from mmf.models.mmbt import MMBT
from mmf.utils.configuration import get_mmf_env, load_yaml
from mmf.utils.file_io import PathManager


class TestModelInterfaces(unittest.TestCase):
    @test_utils.skip_if_no_network
    @test_utils.skip_if_windows
    @test_utils.skip_if_macos
    def test_mmbt_hm_interface(self):
        model = MMBT.from_pretrained("mmbt.hateful_memes.images")
        self._test_model_performance(model)

    @test_utils.skip_if_no_network
    @test_utils.skip_if_windows
    @test_utils.skip_if_macos
    def test_mmbt_hm_interface_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
            self._create_checkpoint_file(tmp.name)

            model = MMBT.from_pretrained(tmp.name, interface=True)
            self._test_model_performance(model)

    def _test_model_performance(self, model):
        result = model.classify(
            "https://i.imgur.com/tEcsk5q.jpg", "look how many people love you"
        )
        self.assertEqual(result["label"], 0)
        np.testing.assert_almost_equal(result["confidence"], 0.9993, decimal=3)
        result = model.classify(
            "https://i.imgur.com/tEcsk5q.jpg", "they have the privilege"
        )
        self.assertEqual(result["label"], 0)
        np.testing.assert_almost_equal(result["confidence"], 0.9777, decimal=1)
        result = model.classify("https://i.imgur.com/tEcsk5q.jpg", "hitler and jews")
        self.assertEqual(result["label"], 1)
        np.testing.assert_almost_equal(result["confidence"], 0.8342, decimal=3)

    def _create_checkpoint_file(self, path):
        home = str(Path.home())
        data_dir = get_mmf_env(key="data_dir")
        model_folder = os.path.join(
            home, data_dir, "models", "mmbt.hateful_memes.images"
        )
        model_file = os.path.join(model_folder, "model.pth")
        config_file = os.path.join(model_folder, "config.yaml")
        config = load_yaml(config_file)
        with PathManager.open(model_file, "rb") as f:
            ckpt = torch.load(f)
        ckpt["config"] = config
        torch.save(ckpt, path)
