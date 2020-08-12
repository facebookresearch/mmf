# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import mmf.modules.fusions as fusions
import torch


class TestModuleFusions(unittest.TestCase):
    def setUp(self):
        bsize = 2
        self.x = [torch.randn(bsize, 10), torch.randn(bsize, 20)]
        self.input_dims = [self.x[0].shape[-1], self.x[1].shape[-1]]
        self.output_dims = 2

    def test_BlockFusion(self):
        fusion = fusions.Block(self.input_dims, self.output_dims, mm_dim=20)
        out = fusion(self.x)
        if torch.cuda.is_available():
            fusion.cuda()
            out = fusion([self.x[0].cuda(), self.x[1].cuda()])
        assert torch.Size([2, 2]) == out.shape

    def test_BlockTucker(self):
        fusion = fusions.BlockTucker(self.input_dims, self.output_dims, mm_dim=20)
        out = fusion(self.x)
        if torch.cuda.is_available():
            fusion.cuda()
            out = fusion([self.x[0].cuda(), self.x[1].cuda()])
        assert torch.Size([2, 2]) == out.shape

    def test_Mutan(self):
        fusion = fusions.Mutan(self.input_dims, self.output_dims, mm_dim=20)
        out = fusion(self.x)
        if torch.cuda.is_available():
            fusion.cuda()
            out = fusion([self.x[0].cuda(), self.x[1].cuda()])
        assert torch.Size([2, 2]) == out.shape

    def test_Tucker(self):
        fusion = fusions.Tucker(self.input_dims, self.output_dims, mm_dim=20)
        out = fusion(self.x)
        if torch.cuda.is_available():
            fusion.cuda()
            out = fusion([self.x[0].cuda(), self.x[1].cuda()])
        assert torch.Size([2, 2]) == out.shape

    def test_MLB(self):
        fusion = fusions.MLB(self.input_dims, self.output_dims, mm_dim=20)
        out = fusion(self.x)
        if torch.cuda.is_available():
            fusion.cuda()
            out = fusion([self.x[0].cuda(), self.x[1].cuda()])
        assert torch.Size([2, 2]) == out.shape

    def test_MFB(self):
        fusion = fusions.MFB(self.input_dims, self.output_dims, mm_dim=20)
        out = fusion(self.x)
        if torch.cuda.is_available():
            fusion.cuda()
            out = fusion([self.x[0].cuda(), self.x[1].cuda()])
        assert torch.Size([2, 2]) == out.shape

    def test_MFH(self):
        fusion = fusions.MFH(self.input_dims, self.output_dims, mm_dim=20)
        out = fusion(self.x)
        if torch.cuda.is_available():
            fusion.cuda()
            out = fusion([self.x[0].cuda(), self.x[1].cuda()])
        assert torch.Size([2, 2]) == out.shape

    def test_MCB(self):
        fusion = fusions.MCB(self.input_dims, self.output_dims, mm_dim=100)
        out = fusion(self.x)
        if torch.cuda.is_available():
            fusion.cuda()
            out = fusion([self.x[0].cuda(), self.x[1].cuda()])
        assert torch.Size([2, 2]) == out.shape

    def test_LinearSum(self):
        fusion = fusions.LinearSum(self.input_dims, self.output_dims, mm_dim=20)
        out = fusion(self.x)
        if torch.cuda.is_available():
            fusion.cuda()
            out = fusion([self.x[0].cuda(), self.x[1].cuda()])
        assert torch.Size([2, 2]) == out.shape

    def test_ConcatMLP(self):
        fusion = fusions.ConcatMLP(self.input_dims, self.output_dims, dimensions=[5, 5])
        out = fusion(self.x)
        if torch.cuda.is_available():
            fusion.cuda()
            out = fusion([self.x[0].cuda(), self.x[1].cuda()])
        assert torch.Size([2, 2]) == out.shape
