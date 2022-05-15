# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.common.sample import Sample
from mmf.models.transformers.heads.utils import build_heads_dict, HeadsDict
from mmf.modules.losses import MMFLoss
from omegaconf import OmegaConf


class TestHeadsDict(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {"type": "mlm", "freeze": False, "vocab_size": 1000, "hidden_size": 768}
        )
        hidden_size = 768
        sample_list = Sample()
        sample_list["targets"] = torch.rand((1, 2))
        sample_list["dataset_type"] = "test"
        sample_list["dataset_name"] = "test_dataset"
        sample_list["is_correct"] = torch.ones((1,), dtype=torch.long)
        self.sample_list = sample_list
        self.model_output = torch.rand(size=(1, 1, hidden_size))
        self.losses = {"test_cls": MMFLoss("logit_bce")}

    def test_constructor_on_dict_confs(self):
        heads_conf = {"test": {"type": "mlp", "loss": "test_cls"}}
        tasks = ["test"]
        heads_dict = build_heads_dict(heads_conf, tasks, self.losses)
        self.assertTrue(isinstance(heads_dict, HeadsDict))

        # test forward
        task = "test"
        head_output = heads_dict.forward(task, self.model_output, self.sample_list)
        self.assertTrue(isinstance(head_output, dict))
        self.assertTrue("losses" in head_output)
        self.assertTrue("test/test_dataset/logit_bce" in head_output["losses"])

    def test_constructor_on_list_confs(self):
        heads_conf = [{"type": "mlp", "loss": "test_cls"}]
        tasks = []
        heads_dict = build_heads_dict(heads_conf, tasks, self.losses)
        self.assertTrue(isinstance(heads_dict, HeadsDict))

        # test forward
        task = None
        head_output = heads_dict.forward(task, self.model_output, self.sample_list)
        self.assertTrue(isinstance(head_output, dict))
        self.assertTrue("losses" in head_output)
        self.assertTrue("test/test_dataset/logit_bce" in head_output["losses"])

    def test_constructor_on_multiple_losses_per_task(self):
        heads_conf = {"test": [{"type": "mlp", "loss": "test_cls"}, {"type": "itm"}]}
        tasks = ["test"]
        heads_dict = build_heads_dict(heads_conf, tasks, self.losses)
        self.assertTrue(isinstance(heads_dict, HeadsDict))

        # test forward
        task = "test"
        head_output = heads_dict.forward(task, self.model_output, self.sample_list)
        self.assertTrue(isinstance(head_output, dict))
        self.assertTrue("losses" in head_output)
        self.assertTrue("test/test_dataset/logit_bce" in head_output["losses"])
        self.assertTrue("itm_loss" in head_output["losses"])

    def test_constructor_on_multiple_tasks(self):
        heads_conf = {
            "test": {"type": "mlp", "loss": "test_cls"},
            "other_task": {"type": "itm"},
            "third_task": {"type": "mlm"},
        }
        tasks = ["test", "other_task"]
        heads_dict = build_heads_dict(heads_conf, tasks, self.losses)
        self.assertTrue(isinstance(heads_dict, HeadsDict))

        # test forward
        task = "other_task"
        head_output = heads_dict.forward(task, self.model_output, self.sample_list)
        self.assertTrue(isinstance(head_output, dict))
        self.assertTrue("losses" in head_output)
        self.assertTrue("test/test_dataset/logit_bce" not in head_output["losses"])
        self.assertTrue("itm_loss" in head_output["losses"])

    def test_constructor_on_multiple_loss_list(self):
        heads_conf = [{"type": "mlp", "loss": "test_cls"}, {"type": "itm"}]
        tasks = []
        heads_dict = build_heads_dict(heads_conf, tasks, self.losses)
        self.assertTrue(isinstance(heads_dict, HeadsDict))

        # test forward
        task = None
        head_output = heads_dict.forward(task, self.model_output, self.sample_list)
        self.assertTrue(isinstance(head_output, dict))
        self.assertTrue("losses" in head_output)
        self.assertTrue("test/test_dataset/logit_bce" in head_output["losses"])
        self.assertTrue("itm_loss" in head_output["losses"])
