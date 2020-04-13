# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.visual_entailment.dataset import VisualEntailmentDataset
from mmf.datasets.builders.vqa2.builder import VQA2Builder


@registry.register_builder("visual_entailment")
class VisualEntailmentBuilder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "visual_entailment"
        self.dataset_class = VisualEntailmentDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/visual_entailment/defaults.yaml"
