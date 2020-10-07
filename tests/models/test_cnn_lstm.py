# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest

import numpy as np
import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.models.cnn_lstm import CNNLSTM
from mmf.utils.configuration import Configuration
from mmf.utils.general import get_mmf_root
from tests.test_utils import dummy_args


class TestModelCNNLSTM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        registry.register("clevr_text_vocab_size", 80)
        registry.register("clevr_num_final_outputs", 32)
        config_path = os.path.join(
            get_mmf_root(),
            "..",
            "projects",
            "others",
            "cnn_lstm",
            "clevr",
            "defaults.yaml",
        )
        config_path = os.path.abspath(config_path)
        args = dummy_args(model="cnn_lstm", dataset="clevr")
        args.opts.append(f"config={config_path}")
        configuration = Configuration(args)
        configuration.config.datasets = "clevr"
        configuration.freeze()
        self.config = configuration.config
        registry.register("config", self.config)

    def test_forward(self):
        model_config = self.config.model_config.cnn_lstm

        cnn_lstm = CNNLSTM(model_config)
        cnn_lstm.build()
        cnn_lstm.init_losses()

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

        np.testing.assert_almost_equal(loss.item(), 19.2635, decimal=4)
        self.assertEqual(scores.size(), torch.Size((1, 32)))
