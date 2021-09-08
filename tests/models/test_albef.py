# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
from mmf.models.albef.vit import AlbefVitEncoder
from omegaconf import OmegaConf
from tests.test_utils import setup_proxy
from torch import nn


class TestAlbefEncoders(unittest.TestCase):
    def setUp(self):
        setup_proxy()

    def _test_init(self, cls, **params):
        encoder = cls.from_params(**params)
        self.assertTrue(isinstance(encoder, nn.Module))

    def test_vision_transformer(self):
        config = OmegaConf.structured(AlbefVitEncoder.Config())
        encoder = AlbefVitEncoder(config)
        x = torch.rand((1, 3, 224, 224))
        output = encoder(x)
        self.assertEqual(output.size(-1), config.out_dim)
