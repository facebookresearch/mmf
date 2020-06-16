# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.mmimdb.dataset import (
    MMIMDbFeaturesDataset,
    MMIMDbImageDataset,
)
from mmf.datasets.builders.vqa2.builder import VQA2Builder


@registry.register_builder("mmimdb")
class MMIMDbBuilder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "mmimdb"
        self.dataset_class = MMIMDbImageDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/mmimdb/defaults.yaml"

    def load(self, config, dataset_type, *args, **kwargs):
        config = config

        if config.use_features:
            self.dataset_class = MMIMDbFeaturesDataset

        self.dataset = super().load(config, dataset_type, *args, **kwargs)

        return self.dataset
