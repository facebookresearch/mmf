# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import mmf.modules.poolers as poolers
import torch


class TestModulePoolers(unittest.TestCase):
    def setUp(self):
        self.k = 2
        self.batch_size = 64
        self.num_tokens = 10
        self.embedding_size = 768
        self.token_len = 10
        self.encoded_layers = [
            torch.randn(self.batch_size, self.token_len, self.embedding_size),
            torch.randn(self.batch_size, self.token_len, self.embedding_size),
            torch.randn(self.batch_size, self.token_len, self.embedding_size),
        ]
        self.pad_mask = torch.randn(self.batch_size, self.token_len)

    def test_AverageConcat(self):
        pool_fn = poolers.AverageConcatLastN(self.k)
        out = pool_fn(self.encoded_layers, self.pad_mask)
        if torch.cuda.is_available():
            pool_fn.cuda()
            out = pool_fn(self.encoded_layers.cuda(), self.pad_mask.cuda())

        assert torch.Size([self.batch_size, self.embedding_size * self.k]) == out.shape

    def test_AverageKFromLast(self):
        pool_fn = poolers.AverageKFromLast(self.k)
        out = pool_fn(self.encoded_layers, self.pad_mask)
        if torch.cuda.is_available():
            pool_fn.cuda()
            out = pool_fn([self.encoded_layers.cuda(), self.pad_mask.cuda()])

        assert torch.Size([self.batch_size, self.embedding_size]) == out.shape

    def test_AverageSumLastK(self):
        pool_fn = poolers.AverageSumLastK(self.k)
        out = pool_fn(self.encoded_layers, self.pad_mask)
        if torch.cuda.is_available():
            pool_fn.cuda()
            out = pool_fn([self.encoded_layers.cuda(), self.pad_mask.cuda()])

        assert torch.Size([self.batch_size, self.embedding_size]) == out.shape
