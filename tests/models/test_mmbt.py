# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import tests.test_utils as test_utils
from mmf.models.mmbt import MMBT
from mmf.modules.encoders import (
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


class TestMMBTTorchscript(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        model_name = "mmbt"
        args = test_utils.dummy_args(model=model_name)
        configuration = Configuration(args)
        config = configuration.get_config()
        model_config = config.model_config[model_name]
        model_config["training_head_type"] = "classification"
        model_config["num_labels"] = 2
        model_config.model = model_name
        self.finetune_model = build_model(model_config)

    def test_load_save_finetune_model(self):
        self.assertTrue(test_utils.verify_torchscript_models(self.finetune_model))

    def test_finetune_model(self):
        model = self.finetune_model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(
                model, vocab_size=BERT_VOCAB_SIZE
            )
        )


class TestMMBTConfig(unittest.TestCase):
    def test_mmbt_from_params(self):
        # default init
        mmbt = MMBT.from_params(
            modal_encoder=ImageEncoderFactory.Config(
                type=ImageEncoderTypes.resnet152,
                params=ResNet152ImageEncoder.Config(pretrained=False),
            ),
            text_encoder=TextEncoderFactory.Config(type=TextEncoderTypes.identity),
        )

        config = OmegaConf.structured(
            MMBT.Config(
                modal_encoder=ImageEncoderFactory.Config(
                    type=ImageEncoderTypes.resnet152,
                    params=ResNet152ImageEncoder.Config(pretrained=False),
                ),
                text_encoder=TextEncoderFactory.Config(type=TextEncoderTypes.identity),
            )
        )
        self.assertIsNotNone(mmbt)
        # Make sure that the config is created from MMBT.Config
        self.assertEqual(mmbt.config, config)

    def test_mmbt_pretrained(self):
        test_utils.setup_proxy()
        mmbt = MMBT.from_params()
        self.assertIsNotNone(mmbt)

    def test_mmbt_directly_from_config(self):
        config = OmegaConf.structured(
            MMBT.Config(
                modal_encoder=ImageEncoderFactory.Config(
                    type=ImageEncoderTypes.resnet152,
                    params=ResNet152ImageEncoder.Config(pretrained=False),
                ),
                text_encoder=TextEncoderFactory.Config(type=TextEncoderTypes.identity),
            )
        )
        mmbt = MMBT(config)
        self.assertIsNotNone(mmbt)
        # Make sure that the config is created from MMBT.Config
        self.assertEqual(mmbt.config, config)
