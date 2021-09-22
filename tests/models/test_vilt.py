# Copyright (c) Facebook, Inc. and its affiliates.

import gc
import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.sample import SampleList
from mmf.utils.build import build_model
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports, teardown_imports
from mmf.utils.general import get_current_device


BERT_VOCAB_SIZE = 30255


class TestViltPretraining(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        model_name = "vilt"
        args = test_utils.dummy_args(
            model=model_name,
            dataset="vqa2",
            config="projects/vilt/configs/vqa2/defaults.yaml",
        )
        configuration = Configuration(args)
        config = configuration.get_config()
        model_config = config.model_config[model_name]
        model_config.model = model_name
        self.pretrain_model = build_model(model_config)

    def tearDown(self):
        teardown_imports()
        del self.pretrain_model
        gc.collect()

    def test_pretrained_model(self):
        sample_list = SampleList()

        sample_list.add_field(
            "input_ids",
            torch.randint(low=0, high=BERT_VOCAB_SIZE, size=(1, 128)).long(),
        )
        sample_list.add_field("input_mask", torch.ones((1, 128)).long())
        sample_list.add_field("segment_ids", torch.zeros(1, 128).long())
        sample_list.add_field("image", torch.rand((1, 3, 224, 224)).float())
        sample_list.add_field("targets", torch.rand((1, 3129)).float())

        self.pretrain_model.eval()
        self.pretrain_model = self.pretrain_model.to(get_current_device())
        sample_list = sample_list.to(get_current_device())

        sample_list.dataset_name = "vqa2"
        sample_list.dataset_type = "test"
        with torch.no_grad():
            model_output = self.pretrain_model(sample_list)

        self.assertTrue("losses" in model_output)
        self.assertTrue("test/vqa2/logit_bce" in model_output["losses"])
