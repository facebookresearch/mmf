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
        loss = output["losses"]["train/logit_bce"]
        accuracy = output["metrics"]["train/accuracy"]

        np.testing.assert_almost_equal(loss.item(), 23.4751, decimal=4)
        np.testing.assert_almost_equal(accuracy.item(), 0)
        self.assertEqual(scores.size(), torch.Size((1, 32)))

        expected_scores = [
            2.2298e-02, -2.4975e-01, -1.1960e-01, -5.0868e-01, -9.3013e-02,
            1.3202e-02, -1.7536e-01, -3.1180e-01,  1.5369e-01,  1.4900e-01,
            1.9006e-01, -1.9457e-01,  1.4924e-02, -1.1032e-01,  1.3777e-01,
            -3.6255e-01, -2.9327e-01,  5.6247e-04, -4.8732e-01,  4.0949e-01,
            -1.1069e-01,  2.9696e-01,  4.1903e-02,  6.7062e-02,  7.0094e-01,
            -1.9898e-01, -2.9502e-03, -3.9040e-01,  1.2218e-01,  3.7895e-02,
            2.4472e-02,  1.7213e-01
        ]
        np.testing.assert_almost_equal(scores[0].tolist(), expected_scores, decimal=5)
