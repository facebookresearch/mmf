# Copyright (c) Facebook, Inc. and its affiliates.

import gc
import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.sample import SampleList
from mmf.models.vilt import ViLTImageEmbedding, ViLTTextEmbedding
from mmf.utils.build import build_model
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports, teardown_imports
from mmf.utils.general import get_current_device
from tests.test_utils import skip_if_old_transformers
from torch import nn


BERT_VOCAB_SIZE = 30255


@skip_if_old_transformers(min_version="4.5.0")
class TestViltEmbeddings(unittest.TestCase):
    def test_vilt_image_embedding(self):
        embedding = ViLTImageEmbedding()
        self.assertTrue(isinstance(embedding, nn.Module))

        image = torch.rand(32, 3, 224, 224)
        output = embedding(image)
        self.assertEqual(output.shape, torch.Size([32, 197, 768]))

    def test_vilt_image_embedding_pretrained(self):
        config = {
            "random_init": False,
            "patch_size": 32,
            "pretrained_model_name": "google/vit-base-patch32-384",
            "image_size": [384, 384],
        }
        embedding = ViLTImageEmbedding(**config)
        self.assertTrue(isinstance(embedding, nn.Module))

        image = torch.rand(32, 3, 384, 384)
        output = embedding(image)
        self.assertEqual(output.shape, torch.Size([32, 145, 768]))

    def test_vilt_text_embedding(self):
        embedding = ViLTTextEmbedding()
        self.assertTrue(isinstance(embedding, nn.Module))

        input_ids = torch.ones(32, 25).long()
        segment_ids = torch.ones(32, 25).long()

        output = embedding(input_ids, segment_ids)
        self.assertEqual(output.shape, torch.Size([32, 25, 768]))


@skip_if_old_transformers(min_version="4.5.0")
class TestViltPretrained(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        model_name = "vilt"
        args = test_utils.dummy_args(model=model_name, dataset="test")
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

        sample_list.dataset_name = "test"
        sample_list.dataset_type = "test"
        with torch.no_grad():
            model_output = self.pretrain_model(sample_list)

        self.assertTrue("losses" in model_output)
        self.assertTrue("test/test/logit_bce" in model_output["losses"])
