# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import operator
import random
import unittest

import mmf.modules.layers as layers
import numpy as np
import torch
from omegaconf import OmegaConf


class TestModuleLayers(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_conv_net(self):
        conv_net = layers.ConvNet(150, 75, 3)

        input_tensor = torch.randn(4, 150, 64, 64)
        output = conv_net(input_tensor)
        expected_size = torch.Size((4, 75, 32, 32))
        self.assertEqual(output.size(), expected_size)
        # Since seed is fix we can check some of tensor values
        np.testing.assert_almost_equal(output[0][0][0][0].item(), 0.149190, decimal=5)
        np.testing.assert_almost_equal(
            output[3][74][31][31].item(), -0.25199, decimal=5
        )

    def test_flatten(self):
        flatten = layers.Flatten()

        # Test 3 dim
        input_tensor = torch.randn(5, 6, 10)
        expected_size = torch.Size((5, 60))
        actual_size = flatten(input_tensor).size()
        self.assertEqual(actual_size, expected_size)

        # Test 1 dim
        input_tensor = torch.randn(5)
        expected_size = torch.Size((5,))
        actual_size = flatten(input_tensor).size()
        self.assertEqual(actual_size, expected_size)

        # Test 6 dim
        size_list = [random.randint(2, 4) for _ in range(7)]
        expected_size = torch.Size(
            (size_list[0], functools.reduce(operator.mul, size_list[1:]))
        )
        input_tensor = torch.randn(*size_list)
        actual_size = flatten(input_tensor).size()
        self.assertEqual(actual_size, expected_size)

    def test_unflatten(self):
        unflatten = layers.UnFlatten()

        # Test 2 dim to 3 dim
        input_tensor = torch.randn(5, 60)
        expected_size = torch.Size((5, 6, 10))
        actual_size = unflatten(input_tensor, sizes=[6, 10]).size()
        self.assertEqual(actual_size, expected_size)

        # Test 1 dim
        input_tensor = torch.randn(5)
        expected_size = torch.Size((5,))
        actual_size = unflatten(input_tensor, sizes=[]).size()
        self.assertEqual(expected_size, actual_size)

    def test_mlp(self):
        mlp = layers.ClassifierLayer("mlp", in_dim=300, out_dim=1)
        self.assertEqual(len(list(mlp.module.layers.children())), 1)
        self.assertEqual(len(list(mlp.parameters())), 2)

        inp = torch.rand(3, 300)

        output = mlp(inp)
        self.assertEqual(output.size(), torch.Size((3, 1)))
        np.testing.assert_almost_equal(
            output.squeeze().tolist(), [0.1949174, 0.4030975, -0.0109139]
        )

        mlp = layers.ClassifierLayer(
            "mlp", in_dim=300, out_dim=1, hidden_dim=150, num_layers=1
        )

        self.assertEqual(len(list(mlp.module.layers.children())), 5)
        self.assertEqual(len(list(mlp.parameters())), 6)

        inp = torch.rand(3, 300)

        output = mlp(inp)
        self.assertEqual(output.size(), torch.Size((3, 1)))
        np.testing.assert_almost_equal(
            output.squeeze().tolist(), [-0.503411, 0.1725615, -0.6833304], decimal=3
        )

    def test_bert_classifier_head(self):
        config = {}
        config["hidden_size"] = 768
        config["hidden_act"] = "gelu"
        config["layer_norm_eps"] = 1e-12
        config["hidden_dropout_prob"] = 0.1
        config = OmegaConf.create(config)
        clf = layers.ClassifierLayer("bert", 768, 1, config=config)
        self.assertEqual(len(list(clf.module.children())), 3)
        self.assertEqual(len(list(clf.parameters())), 6)

        inp = torch.rand(3, 768)

        output = clf(inp)
        self.assertEqual(output.size(), torch.Size((3, 1)))
        np.testing.assert_almost_equal(
            output.squeeze().tolist(), [0.5452202, -0.0437842, -0.377468], decimal=3
        )
