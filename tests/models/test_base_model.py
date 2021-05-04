# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from mmf.common.sample import SampleList
from mmf.models.base_model import BaseModel
from tests.test_utils import compare_tensors


class LocalTestModelWithForwardLoss(BaseModel):
    def forward(self, *args, **kwargs):
        return {"losses": {"x": torch.tensor(1.0)}}


class LocalTestModelWithNoLoss(BaseModel):
    def forward(self, *args, **kwargs):
        return {}


class LocalTestModelWithLossAttribute(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = lambda x, y: {"x": torch.tensor(2.0)}

    def forward(self, *args, **kwargs):
        return {}


class TestBaseModel(unittest.TestCase):
    def test_forward_loss(self):
        sample_list = SampleList()
        sample_list.add_field("x", torch.tensor(1))
        model = LocalTestModelWithForwardLoss({})
        with torch.no_grad():
            output = model(sample_list)
        self.assertTrue("losses" in output)
        self.assertTrue(compare_tensors(output["losses"]["x"], torch.tensor(1.0)))

        model = LocalTestModelWithLossAttribute({})
        with torch.no_grad():
            output = model(sample_list)
        self.assertTrue("losses" in output)
        self.assertTrue(compare_tensors(output["losses"]["x"], torch.tensor(2.0)))

        model = LocalTestModelWithNoLoss({})
        with torch.no_grad():
            output = model(sample_list)
        self.assertTrue("losses" in output)
        self.assertEqual(output["losses"], {})
