# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import tests.test_utils as test_utils
from mmf.models.mmf_transformer import MMFTransformer, MMFTransformerModalityConfig
from mmf.modules.encoders import (
    IdentityEncoder,
    ImageEncoderFactory,
    ImageEncoderTypes,
    ResNet152ImageEncoder,
    TextEncoderFactory,
    TextEncoderTypes,
)
from mmf.utils.build import build_model
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports
from omegaconf import OmegaConf


BERT_VOCAB_SIZE = 30255
ROBERTA_VOCAB_SIZE = 50265
XLM_ROBERTA_VOCAB_SIZE = 250002


class TestMMFTransformerTorchscript(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        self.model_name = "mmf_transformer"
        args = test_utils.dummy_args(model=self.model_name)
        configuration = Configuration(args)
        self.config = configuration.get_config()
        self.config.model_config[self.model_name].model = self.model_name
        self.finetune_model = build_model(self.config.model_config[self.model_name])

    def test_load_save_finetune_model(self):
        self.assertTrue(test_utils.verify_torchscript_models(self.finetune_model))

    def test_finetune_bert_base(self):
        model = self.finetune_model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(
                model, vocab_size=BERT_VOCAB_SIZE
            )
        )

    def test_finetune_roberta_base(self):
        self.config.model_config[self.model_name]["transformer_base"] = "roberta-base"
        model = build_model(self.config.model_config[self.model_name])
        model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(
                model, vocab_size=ROBERTA_VOCAB_SIZE
            )
        )

    @test_utils.skip_if_no_network
    def test_finetune_xlmr_base(self):
        self.config.model_config[self.model_name][
            "transformer_base"
        ] = "xlm-roberta-base"
        model = build_model(self.config.model_config[self.model_name])
        model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(
                model, vocab_size=XLM_ROBERTA_VOCAB_SIZE
            )
        )


class TestMMFTransformerConfig(unittest.TestCase):
    def setUp(self):
        setup_imports()

    def test_mmft_from_params(self):
        modalities_config = [
            MMFTransformerModalityConfig(
                type="image",
                key="image",
                embedding_dim=256,
                position_dim=1,
                segment_id=0,
                encoder=IdentityEncoder.Config(),
            ),
            MMFTransformerModalityConfig(
                type="text",
                key="text",
                embedding_dim=768,
                position_dim=512,
                segment_id=1,
                encoder=IdentityEncoder.Config(),
            ),
        ]
        mmft = MMFTransformer.from_params(modalities=modalities_config, num_labels=2)
        mmft.build()

        config = OmegaConf.structured(
            MMFTransformer.Config(modalities=modalities_config, num_labels=2)
        )
        self.assertIsNotNone(mmft)
        self.assertEqual(mmft.config, config)

    def test_mmf_from_params_encoder_factory(self):
        modalities_config = [
            MMFTransformerModalityConfig(
                type="image",
                key="image",
                embedding_dim=256,
                position_dim=1,
                segment_id=0,
                encoder=ImageEncoderFactory.Config(type=ImageEncoderTypes.identity),
            ),
            MMFTransformerModalityConfig(
                type="text",
                key="text",
                embedding_dim=756,
                position_dim=512,
                segment_id=0,
                encoder=TextEncoderFactory.Config(type=TextEncoderTypes.identity),
            ),
        ]
        mmft = MMFTransformer.from_params(modalities=modalities_config, num_labels=2)
        mmft.build()

        config = OmegaConf.structured(
            MMFTransformer.Config(modalities=modalities_config, num_labels=2)
        )
        self.assertIsNotNone(mmft)
        self.assertEqual(mmft.config, config)

    def test_mmft_pretrained(self):
        mmft = MMFTransformer.from_params(num_labels=2)
        self.assertIsNotNone(mmft)

    def test_mmft_from_build_model(self):
        modalities_config = [
            MMFTransformerModalityConfig(
                type="image",
                key="image",
                embedding_dim=256,
                position_dim=1,
                segment_id=0,
                encoder=ImageEncoderFactory.Config(
                    type=ImageEncoderTypes.resnet152,
                    params=ResNet152ImageEncoder.Config(pretrained=False),
                ),
            ),
            MMFTransformerModalityConfig(
                type="text",
                key="text",
                embedding_dim=756,
                position_dim=512,
                segment_id=1,
                encoder=TextEncoderFactory.Config(type=TextEncoderTypes.identity),
            ),
        ]
        config = MMFTransformer.Config(modalities=modalities_config, num_labels=2)
        mmft = build_model(config)
        self.assertIsNotNone(mmft)
