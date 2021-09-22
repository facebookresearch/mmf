# Copyright (c) Facebook, Inc. and its affiliates.

import tempfile
import unittest

import torch
from mmf.modules import encoders
from omegaconf import OmegaConf
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

    def test_transformer_encoder_forward(self):
        encoder = encoders.TransformerEncoder.from_params()
        self.assertEqual(encoder.embeddings.word_embeddings.weight.size(1), 768)
        self.assertEqual(encoder.embeddings.word_embeddings.weight.size(0), 30522)
        text_ids = torch.randint(
            encoder.embeddings.word_embeddings.weight.size(0), (2, 16)
        )
        text_embeddings = encoder(text_ids)

        text_embeddings_cls = encoder(text_ids)
        self.assertEqual(text_embeddings_cls.dim(), 2)
        self.assertEqual(list(text_embeddings_cls.size()), [2, 768])

        text_embeddings = encoder(text_ids, return_sequence=True)
        self.assertEqual(text_embeddings.dim(), 3)
        self.assertEqual(list(text_embeddings.size()), [2, 16, 768])

    def test_r2plus1d18_video_encoder(self):
        config = OmegaConf.structured(
            encoders.R2Plus1D18VideoEncoder.Config(pretrained=False)
        )
        encoder = encoders.R2Plus1D18VideoEncoder(config)
        x = torch.rand((1, 3, 16, 112, 112))
        output = encoder(x)
        self.assertEqual(output.size(-1), config.out_dim)

    def test_resnet18_audio_encoder(self):
        config = OmegaConf.structured(encoders.ResNet18AudioEncoder.Config())
        encoder = encoders.ResNet18AudioEncoder(config)
        x = torch.rand((1, 1, 4778, 224))
        output = encoder(x)
        self.assertEqual(output.size(-1), config.out_dim)

    def test_vilt_encoder(self):
        from omegaconf import open_dict

        config = OmegaConf.structured(encoders.ViTEncoder.Config())
        with open_dict(config):
            config.update(
                {
                    "layer_norm_eps": 0.0001,
                    "hidden_size": 768,
                    "num_hidden_layers": 2,
                    "do_patch_embeddings": False,
                    "add_pooling_layer": False,
                    "out_dim": 768,
                }
            )
        encoder = encoders.ViTEncoder(config)
        x = torch.rand(32, 197, 768)
        output, _ = encoder(x)
        self.assertEqual(output.size(-1), config.out_dim)

    def test_vilt_image_embedding(self):
        from mmf.common.sample import SampleList

        config = OmegaConf.structured(encoders.VILTImageEmbedding.Config())
        encoder = encoders.VILTImageEmbedding(config)
        self.assertTrue(isinstance(encoder, nn.Module))

        sample_list = SampleList({"image": torch.rand(32, 3, 224, 224)})
        output = encoder(sample_list)
        self.assertEqual(output.shape, torch.Size([32, 197, 768]))

    def test_vilt_text_embedding(self):
        from mmf.common.sample import SampleList

        config = OmegaConf.structured(encoders.VILTTextEmbedding.Config())

        encoder = encoders.VILTTextEmbedding(config)
        self.assertTrue(isinstance(encoder, nn.Module))

        sample_list = SampleList(
            {
                "input_ids": torch.ones(32, 25).long(),
                "segment_ids": torch.ones(32, 25).long(),
            }
        )
        output = encoder(sample_list)
        self.assertEqual(output.shape, torch.Size([32, 25, 768]))
