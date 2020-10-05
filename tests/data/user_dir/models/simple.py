# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from tests.test_utils import SimpleModel


@registry.register_model("simple")
class CustomSimpleModel(SimpleModel):
    @classmethod
    def config_path(cls):
        return "configs/simple.yaml"

    def forward(self, sample_list):
        return {"scores": self.classifier(sample_list.input)}
