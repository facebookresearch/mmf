# Copyright (c) Facebook, Inc. and its affiliates.

import tempfile
import unittest

import torch
from mmf.modules import encoders
from omegaconf import OmegaConf
from tests.test_utils import (
    setup_proxy,
    skip_if_no_pytorchvideo,
    skip_if_old_transformers,
)
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

    @skip_if_old_transformers(min_version="4.5.0")
    def test_vit_encoder(self):
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

    @skip_if_no_pytorchvideo
    def test_pytorchvideo_slowfast_r50_encoder(self):
        # instantiate video encoder from pytorchvideo
        # default model is slowfast_r50
        config = OmegaConf.structured(encoders.PytorchVideoEncoder.Config())
        encoder = encoders.PytorchVideoEncoder(config)
        fast = torch.rand((1, 3, 32, 224, 224))
        slow = torch.rand((1, 3, 8, 224, 224))
        output = encoder([slow, fast])
        # check output tensor is the expected feature dim size
        # (bs, feature_dim)
        self.assertEqual(output.size(1), 2304)

    @skip_if_no_pytorchvideo
    def test_mvit_encoder(self):
        config = {
            "name": "pytorchvideo",
            "model_name": "mvit_base_32x3",
            "random_init": True,
            "drop_last_n_layers": 0,
            "pooler_name": "cls",
            "spatial_size": 224,
            "temporal_size": 8,
            "head": None,
            "embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
            "atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
            "pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
            "pool_kv_stride_adaptive": [1, 8, 8],
            "pool_kvq_kernel": [3, 3, 3],
        }
        # test bert cls pooler
        encoder = encoders.PytorchVideoEncoder(OmegaConf.create(config))
        x = torch.rand((1, 3, 8, 224, 224))
        output = encoder(x)
        # check output tensor is the expected feature dim size
        # based on pooled attention configs
        # for more details consult https://arxiv.org/pdf/2104.11227
        # and https://github.com/facebookresearch/pytorchvideo/
        # (bs, num_features, feature_dim)
        self.assertEqual(output.shape, torch.Size([1, 768]))

        # test avg pooler
        encoder = encoders.PytorchVideoEncoder(
            OmegaConf.create(dict(config, pooler_name="avg"))
        )
        output = encoder(x)
        self.assertEqual(output.shape, torch.Size([1, 768]))

        # test no pooling
        encoder = encoders.PytorchVideoEncoder(
            OmegaConf.create(dict(config, pooler_name="identity"))
        )
        output = encoder(x)
        # (bs, num_features, feature_dim)
        self.assertEqual(output.shape, torch.Size([1, 197, 768]))
