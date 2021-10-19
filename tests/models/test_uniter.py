# Copyright (c) Facebook, Inc. and its affiliates.
import gc
import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.sample import SampleList
from mmf.models.uniter import UniterImageEmbeddings, UniterModelBase
from mmf.utils.build import build_model
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports, teardown_imports
from mmf.utils.general import get_current_device
from omegaconf import OmegaConf


class TestUniterImageEmbeddings(unittest.TestCase):
    def test_forward_has_correct_output_dim(self):
        bs = 32
        num_feat = 100
        config = OmegaConf.create({"img_dim": 1024, "hidden_size": 256, "pos_dim": 7})
        embedding = UniterImageEmbeddings(config)
        img_feat = torch.rand((bs, num_feat, config["img_dim"]))
        img_pos_feat = torch.rand((bs, num_feat, config["pos_dim"]))
        type_embeddings = torch.ones((bs, num_feat, 1), dtype=torch.long)

        output = embedding(img_feat, img_pos_feat, type_embeddings, img_masks=None)
        self.assertEquals(list(output.shape), [32, 100, 256])


class TestUniterModelBase(unittest.TestCase):
    def tearDown(self):
        del self.model
        gc.collect()

    def test_pretrained_model(self):
        img_dim = 1024
        config = OmegaConf.create({"image_embeddings": {"img_dim": img_dim}})
        self.model = UniterModelBase(config)

        self.model.eval()
        self.model = self.model.to(get_current_device())

        bs = 8
        num_feats = 100
        max_sentence_len = 25
        pos_dim = 7
        input_ids = torch.ones((bs, max_sentence_len), dtype=torch.long)
        img_feat = torch.rand((bs, num_feats, img_dim))
        img_pos_feat = torch.rand((bs, num_feats, pos_dim))
        position_ids = torch.arange(
            0, input_ids.size(1), dtype=torch.long, device=img_feat.device
        ).unsqueeze(0)
        attention_mask = torch.ones((bs, max_sentence_len + num_feats))

        with torch.no_grad():
            model_output = self.model(
                input_ids, position_ids, img_feat, img_pos_feat, attention_mask
            )

        self.assertEqual(model_output.shape, torch.Size([8, 125, 768]))


class TestUniterModel(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        model_name = "uniter"
        args = test_utils.dummy_args(model=model_name, dataset="vqa2")
        configuration = Configuration(args)
        config = configuration.get_config()
        model_config = config.model_config[model_name]
        model_config.model = model_name
        model_config.losses = {"vqa2": "logit_bce"}
        self.pretrain_model = build_model(model_config)

    def tearDown(self):
        teardown_imports()
        del self.pretrain_model
        gc.collect()

    def test_pretrained_model(self):
        bs = 8
        num_feats = 100
        max_sentence_len = 25
        img_dim = 2048
        vqa_cls_dim = 3129
        input_ids = torch.ones((bs, max_sentence_len), dtype=torch.long)
        input_mask = torch.ones((bs, max_sentence_len), dtype=torch.long)
        img_feat = torch.rand((bs, num_feats, img_dim))

        max_features = torch.ones((bs, num_feats)) * num_feats
        bbox = torch.randint(50, 200, (bs, num_feats, 4)).float()
        image_height = torch.randint(100, 300, (bs,))
        image_width = torch.randint(100, 300, (bs,))
        image_info = {
            "max_features": max_features,
            "bbox": bbox,
            "image_height": image_height,
            "image_width": image_width,
        }
        targets = torch.rand((bs, vqa_cls_dim))

        sample_list = SampleList()
        sample_list.add_field("input_ids", input_ids)
        sample_list.add_field("image_feature_0", img_feat)
        sample_list.add_field("input_mask", input_mask)
        sample_list.add_field("image_info_0", image_info)
        sample_list.add_field("targets", targets)

        self.pretrain_model.eval()
        self.pretrain_model = self.pretrain_model.to(get_current_device())
        sample_list = sample_list.to(get_current_device())

        sample_list.dataset_name = "vqa2"
        sample_list.dataset_type = "test"
        with torch.no_grad():
            model_output = self.pretrain_model(sample_list)

        self.assertTrue("losses" in model_output)
        self.assertTrue("test/vqa2/logit_bce" in model_output["losses"])
