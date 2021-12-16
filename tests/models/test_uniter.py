# Copyright (c) Facebook, Inc. and its affiliates.
import gc
import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.sample import SampleList
from mmf.models.uniter import (
    UNITERForClassification,
    UNITERForPretraining,
    UNITERImageEmbeddings,
    UNITERModelBase,
)
from mmf.utils.build import build_model
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports, teardown_imports
from mmf.utils.general import get_current_device
from omegaconf import OmegaConf


class TestUNITERImageEmbeddings(unittest.TestCase):
    def setUp(self):
        bs = 32
        num_feat = 100
        self.config = OmegaConf.create(
            {"img_dim": 1024, "hidden_size": 256, "pos_dim": 7}
        )
        self.img_feat = torch.rand((bs, num_feat, self.config["img_dim"]))
        self.img_pos_feat = torch.rand((bs, num_feat, self.config["pos_dim"]))
        self.type_embeddings = torch.ones((bs, num_feat, 1), dtype=torch.long)

    def test_forward(self):
        embedding = UNITERImageEmbeddings(**self.config)
        output = embedding(
            self.img_feat, self.img_pos_feat, self.type_embeddings, img_masks=None
        )
        self.assertEquals(list(output.shape), [32, 100, 256])


class TestUNITERModelBase(unittest.TestCase):
    def test_pretrained_model(self):
        img_dim = 1024
        model = UNITERModelBase(img_dim=img_dim)

        device = get_current_device()
        model.eval()
        model = model.to(device)

        bs = 8
        num_feats = 100
        max_sentence_len = 25
        pos_dim = 7
        input_ids = torch.ones((bs, max_sentence_len), dtype=torch.long).to(device)
        img_feat = torch.rand((bs, num_feats, img_dim)).to(device)
        img_pos_feat = torch.rand((bs, num_feats, pos_dim)).to(device)
        position_ids = torch.arange(
            0, input_ids.size(1), dtype=torch.long, device=device
        ).unsqueeze(0)
        attention_mask = torch.ones((bs, max_sentence_len + num_feats)).to(device)

        with torch.no_grad():
            model_output = model(
                input_ids, position_ids, img_feat, img_pos_feat, attention_mask
            ).final_layer

        self.assertEqual(model_output.shape, torch.Size([8, 125, 768]))


class TestUniterWithHeads(unittest.TestCase):
    def _get_sample_list(self):
        bs = 8
        num_feats = 100
        max_sentence_len = 25
        img_dim = 2048
        cls_dim = 3129
        input_ids = torch.ones((bs, max_sentence_len), dtype=torch.long)
        input_mask = torch.ones((bs, max_sentence_len), dtype=torch.long)
        image_feat = torch.rand((bs, num_feats, img_dim))
        position_ids = (
            torch.arange(
                0, max_sentence_len, dtype=torch.long, device=image_feat.device
            )
            .unsqueeze(0)
            .expand(bs, -1)
        )
        img_pos_feat = torch.rand((bs, num_feats, 7))
        attention_mask = torch.zeros(
            (bs, max_sentence_len + num_feats), dtype=torch.long
        )
        image_mask = torch.zeros((bs, num_feats), dtype=torch.long)
        targets = torch.rand((bs, cls_dim))

        sample_list = SampleList()
        sample_list.add_field("input_ids", input_ids)
        sample_list.add_field("input_mask", input_mask)
        sample_list.add_field("image_feat", image_feat)
        sample_list.add_field("img_pos_feat", img_pos_feat)
        sample_list.add_field("attention_mask", attention_mask)
        sample_list.add_field("image_mask", image_mask)
        sample_list.add_field("targets", targets)
        sample_list.add_field("dataset_name", "test")
        sample_list.add_field("dataset_type", "test")
        sample_list.add_field("position_ids", position_ids)
        sample_list.to(get_current_device())

        return sample_list

    def test_uniter_for_classification(self):
        heads = {"test": {"type": "mlp", "num_labels": 3129}}
        tasks = "test"
        losses = {"test": "logit_bce"}
        model = UNITERForClassification(
            head_configs=heads, loss_configs=losses, tasks=tasks
        )

        model.eval()
        model = model.to(get_current_device())
        sample_list = self._get_sample_list()

        with torch.no_grad():
            model_output = model(sample_list)

        self.assertTrue("losses" in model_output)
        self.assertTrue("test/test/logit_bce" in model_output["losses"])

    def _enhance_sample_list_for_pretraining(self, sample_list):
        bs = sample_list["input_ids"].size(0)
        sentence_len = sample_list["input_ids"].size(1)

        is_correct = torch.ones((bs,), dtype=torch.long)
        lm_label_ids = torch.zeros((bs, sentence_len), dtype=torch.long)
        input_ids_masked = sample_list["input_ids"]
        num_feat = sample_list["image_feat"].size(1)
        cls_dim = 1601
        image_info = {"cls_prob": torch.rand((bs, num_feat, cls_dim))}
        sample_list.add_field("is_correct", is_correct)
        sample_list.add_field("task", "mlm")
        sample_list.add_field("lm_label_ids", lm_label_ids)
        sample_list.add_field("input_ids_masked", input_ids_masked)
        sample_list.add_field("image_info_0", image_info)
        sample_list.to(get_current_device())

    def test_uniter_for_pretraining(self):
        # UNITER pretraining has 5 pretraining tasks,
        # we have one unique head for each, and in each
        # forward pass we train on a different task.
        # In this test we try running a forward pass
        # through each head.
        heads = {
            "mlm": {"type": "mlm"},
            "itm": {"type": "itm"},
            "mrc": {"type": "mrc"},
            "mrfr": {"type": "mrfr"},
            "wra": {"type": "wra"},
        }
        tasks = "mlm,itm,mrc,mrfr,wra"
        mask_probability = 0.15
        model = UNITERForPretraining(
            head_configs=heads, tasks=tasks, mask_probability=mask_probability
        )
        model.eval()
        model = model.to(get_current_device())
        sample_list = self._get_sample_list()
        self._enhance_sample_list_for_pretraining(sample_list)

        expected_loss_names = {
            "mlm": "masked_lm_loss",
            "itm": "itm_loss",
            "mrc": "mrc_loss",
            "mrfr": "mrfr_loss",
            "wra": "wra_loss",
        }

        for task_name, loss_name in expected_loss_names.items():
            sample_list["task"] = task_name
            with torch.no_grad():
                model_output = model(sample_list)

            self.assertTrue("losses" in model_output)
            self.assertTrue(loss_name in model_output["losses"])


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
        model_config.do_pretraining = False
        model_config.tasks = "vqa2"
        classification_config_dict = {
            "do_pretraining": False,
            "tasks": "vqa2",
            "heads": {"vqa2": {"type": "mlp", "num_labels": 3129}},
            "losses": {"vqa2": "logit_bce"},
        }
        classification_config = OmegaConf.create(
            {**model_config, **classification_config_dict}
        )

        pretraining_config_dict = {
            "do_pretraining": True,
            "tasks": "wra",
            "heads": {"wra": {"type": "wra"}},
        }
        pretraining_config = OmegaConf.create(
            {**model_config, **pretraining_config_dict}
        )

        self.model_for_classification = build_model(classification_config)
        self.model_for_pretraining = build_model(pretraining_config)

    def tearDown(self):
        teardown_imports()
        del self.model_for_classification
        del self.model_for_pretraining
        gc.collect()

    def _get_sample_list(self):
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
        is_correct = torch.ones((bs,), dtype=torch.long)

        sample_list = SampleList()
        sample_list.add_field("input_ids", input_ids)
        sample_list.add_field("image_feature_0", img_feat)
        sample_list.add_field("input_mask", input_mask)
        sample_list.add_field("image_info_0", image_info)
        sample_list.add_field("targets", targets)
        sample_list.add_field("is_correct", is_correct)
        sample_list = sample_list.to(get_current_device())
        return sample_list

    def test_uniter_for_classification(self):
        self.model_for_classification.eval()
        self.model_for_classification = self.model_for_classification.to(
            get_current_device()
        )
        sample_list = self._get_sample_list()

        sample_list.dataset_name = "vqa2"
        sample_list.dataset_type = "test"
        with torch.no_grad():
            model_output = self.model_for_classification(sample_list)

        self.assertTrue("losses" in model_output)
        self.assertTrue("test/vqa2/logit_bce" in model_output["losses"])

    def test_uniter_for_pretraining(self):
        self.model_for_pretraining.eval()
        self.model_for_pretraining = self.model_for_pretraining.to(get_current_device())
        sample_list = self._get_sample_list()
        sample_list["tasks"] = "wra"

        sample_list.dataset_name = "vqa2"
        sample_list.dataset_type = "test"
        with torch.no_grad():
            model_output = self.model_for_pretraining(sample_list)

        self.assertTrue("losses" in model_output)
        self.assertTrue("wra_loss" in model_output["losses"])
