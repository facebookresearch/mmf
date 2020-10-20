# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.sample import Sample, SampleList
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
        self.finetune_model.eval()
        test_sample = Sample()
        test_sample.input_ids = torch.randint(low=0, high=30255, size=(128,)).long()
        test_sample.input_mask = torch.ones(128).long()
        test_sample.segment_ids = torch.zeros(128).long()
        test_sample.image = torch.rand((3, 300, 300)).float()
        test_sample_list = SampleList([test_sample.copy()])

        with torch.no_grad():
            model_output = self.finetune_model.model(test_sample_list)

        test_sample_list = SampleList([test_sample])
        script_model = torch.jit.script(self.finetune_model.model)
        with torch.no_grad():
            script_output = script_model(test_sample_list)

        self.assertTrue(torch.equal(model_output["scores"], script_output["scores"]))

    def test_modal_end_token(self):
        self.finetune_model.eval()

        # Suppose 0 for <cls>, 1 for <pad> 2 for <sep>
        CLS = 0
        PAD = 1
        SEP = 2
        size = 128

        input_ids = torch.randint(low=0, high=30255, size=(size,)).long()
        input_mask = torch.ones(size).long()

        input_ids[0] = CLS
        length = torch.randint(low=2, high=size - 1, size=(1,))
        input_ids[length] = SEP
        input_ids[length + 1 :] = PAD
        input_mask[length + 1 :] = 0

        test_sample = Sample()
        test_sample.input_ids = input_ids.clone()
        test_sample.input_mask = input_mask.clone()
        test_sample.segment_ids = torch.zeros(size).long()
        test_sample.image = torch.rand((3, 300, 300)).float()
        test_sample_list = SampleList([test_sample])

        mmbt_base = self.finetune_model.model.bert
        with torch.no_grad():
            actual_modal_end_token = mmbt_base.extract_modal_end_token(test_sample_list)

        expected_modal_end_token = torch.zeros([1]).fill_(SEP).long()
        self.assertTrue(torch.equal(actual_modal_end_token, expected_modal_end_token))
        self.assertTrue(torch.equal(test_sample_list.input_ids[0, :-1], input_ids[1:]))
        self.assertTrue(
            torch.equal(test_sample_list.input_mask[0, :-1], input_mask[1:])
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
