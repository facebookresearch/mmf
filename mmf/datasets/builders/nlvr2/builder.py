# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.nlvr2.dataset import NLVR2Dataset
from mmf.datasets.builders.vqa2.builder import VQA2Builder


@registry.register_builder("nlvr2")
class NLVR2Builder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "nlvr2"
        self.dataset_class = NLVR2Dataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/nlvr2/defaults.yaml"
