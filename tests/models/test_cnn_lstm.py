# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
import random
import os
import numpy as np

from pythia.models.cnn_lstm import CNNLSTM
from pythia.common.registry import registry
from pythia.common.sample import Sample, SampleList
from pythia.utils.configuration import ConfigNode, Configuration
from pythia.utils.general import get_pythia_root


class TestModelCNNLSTM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        registry.register("clevr_text_vocab_size", 80)
        registry.register("clevr_num_final_outputs", 32)
        config_path = os.path.join(
            get_pythia_root(), "..", "configs", "vqa", "clevr", "cnn_lstm.yml"
        )
        config_path = os.path.abspath(config_path)
        configuration = Configuration(config_path)
        configuration.config["datasets"] = "clevr"
        configuration.freeze()
        self.config = configuration.config
        registry.register("config", self.config)

    def test_forward(self):
        model_config = self.config.model_attributes.cnn_lstm

        cnn_lstm = CNNLSTM(model_config)
        cnn_lstm.build()
        cnn_lstm.init_losses_and_metrics()

        self.assertTrue(isinstance(cnn_lstm, torch.nn.Module))

        test_sample = Sample()
        test_sample.text = torch.randint(1, 79, (10,), dtype=torch.long)
        test_sample.image = torch.randn(3, 320, 480)
        test_sample.targets = torch.randn(32)

        test_sample_list = SampleList([test_sample])
        test_sample_list.dataset_type = "train"
        test_sample_list.dataset_name = "clevr"
        output = cnn_lstm(test_sample_list)

        scores = output["scores"]
        loss = output["losses"]["train/clevr/logit_bce"]
        accuracy = output["metrics"]["train/clevr/accuracy"]

        np.testing.assert_almost_equal(loss.item(), 19.2635, decimal=4)
        np.testing.assert_almost_equal(accuracy.item(), 0)
        self.assertEqual(scores.size(), torch.Size((1, 32)))

        expected_scores = [
            -0.7598285675048828, -0.07029829174280167, -0.20382611453533173, -0.06990239024162292,
            0.7965695858001709, 0.4730074405670166, -0.30569902062416077, 0.4244227707386017,
            0.6511023044586182, 0.2480515092611313, -0.5087617635726929, -0.7675772905349731,
            0.4361543357372284, 0.0018743239343166351, 0.6774630546569824, 0.30618518590927124,
            -0.398895800113678, -0.13120117783546448, -0.4433199465274811, -0.25969570875167847,
            0.6798790097236633, -0.34090861678123474, 0.0384102463722229, 0.2484571784734726,
            0.0456063412129879, -0.428459107875824, -0.026385333389043808, -0.1570669412612915,
            -0.2377825379371643, 0.3231588304042816, 0.21098048985004425, -0.712349534034729
        ]

        np.testing.assert_almost_equal(scores[0].tolist(), expected_scores, decimal=5)
