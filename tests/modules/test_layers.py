# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
import random
import operator
import functools
import numpy as np

import pythia.modules.layers as layers


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
        np.testing.assert_almost_equal(output[3][74][31][31].item(), -0.25199, decimal=5)


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
        expected_size = torch.Size((size_list[0], functools.reduce(operator.mul, size_list[1:])))
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
