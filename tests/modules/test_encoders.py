# Copyright (c) Facebook, Inc. and its affiliates.

import tempfile
import unittest

from mmf.modules import encoders
from tests.test_utils import setup_proxy
from torch import nn


class TestEncoders(unittest.TestCase):
    def setUp(self):
        setup_proxy()

    def _test_init(self, cls, **params):
        encoder = cls.from_params(**params)
        self.assertTrue(isinstance(encoder, nn.Module))

    def test_finetune_faster_rcnn_fpn_fc7(self):
        # Add tempfile dir so that the encoder downloads data automatically for testing
        self._test_init(
            encoders.FinetuneFasterRcnnFpnFc7,
            in_dim=2048,
            model_data_dir=tempfile.TemporaryDirectory().name,
        )

    def test_resnet152_image_encoder(self):
        self._test_init(encoders.ResNet152ImageEncoder)

    def test_text_embedding_encoder(self):
        embedding_params = {
            "type": "projection",
            "params": {"module": "linear", "in_dim": 756, "out_dim": 756},
        }
        self._test_init(
            encoders.TextEmbeddingEncoder,
            operator="sum",
            embedding_params=embedding_params,
        )

    def test_transformer_encoder(self):
        self._test_init(encoders.TransformerEncoder)

    def test_multimodal_encoder_base(self):
        self._test_init(encoders.MultiModalEncoderBase)

    def test_identity(self):
        self._test_init(encoders.IdentityEncoder, in_dim=256)
