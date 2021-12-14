# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import mmf.modules.poolers as poolers
import torch
from mmf.utils.general import get_current_device


class TestModulePoolers(unittest.TestCase):
    def setUp(self):
        self.k = 2
        self.batch_size = 64
        self.num_tokens = 10
        self.embedding_size = 768
        self.token_len = 10
        self.device = get_current_device()
        self.encoded_layers = [
            torch.randn(self.batch_size, self.token_len, self.embedding_size).to(
                self.device
            )
            for _ in range(3)
        ]
        self.pad_mask = torch.randn(self.batch_size, self.token_len).to(self.device)

    def test_AverageConcat(self):
        pool_fn = poolers.AverageConcatLastN(self.k).to(self.device)
        out = pool_fn(self.encoded_layers, self.pad_mask)
<<<<<<< HEAD
        if torch.cuda.is_available():
            pool_fn.cuda()
            out = pool_fn([e.cuda() for e in self.encoded_layers], self.pad_mask.cuda())


=======
>>>>>>> upstream/main

        assert torch.Size([self.batch_size, self.embedding_size * self.k]) == out.shape

    def test_AverageKFromLast(self):
        pool_fn = poolers.AverageKFromLast(self.k).to(self.device)
        out = pool_fn(self.encoded_layers, self.pad_mask)
<<<<<<< HEAD
        if torch.cuda.is_available():
            pool_fn.cuda()
            out = pool_fn([e.cuda() for e in self.encoded_layers], self.pad_mask.cuda())


=======
>>>>>>> upstream/main

        assert torch.Size([self.batch_size, self.embedding_size]) == out.shape

    def test_AverageSumLastK(self):
        pool_fn = poolers.AverageSumLastK(self.k).to(self.device)
        out = pool_fn(self.encoded_layers, self.pad_mask)
<<<<<<< HEAD
        if torch.cuda.is_available():
            pool_fn.cuda()
            out = pool_fn([e.cuda() for e in self.encoded_layers], self.pad_mask.cuda())


=======
>>>>>>> upstream/main

        assert torch.Size([self.batch_size, self.embedding_size]) == out.shape
