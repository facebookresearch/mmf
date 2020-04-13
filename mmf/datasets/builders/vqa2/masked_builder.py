# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.vqa2.builder import VQA2Builder
from mmf.datasets.builders.vqa2.masked_dataset import MaskedVQA2Dataset


@registry.register_builder("masked_vqa2")
class MaskedVQA2Builder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_vqa2"
        self.dataset_class = MaskedVQA2Dataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/vqa2/masked.yaml"
