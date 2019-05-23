# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pythia.common.registry import registry
from pythia.tasks.vqa.vqa2 import VQA2Builder

from .dataset import COCODataset


@registry.register_builder("coco")
class COCOBuilder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "coco"
        self.set_dataset_class(COCODataset)

    def update_registry_for_model(self, config):
        registry.register(
            self.dataset_name + "_text_vocab_size",
            self.dataset.text_processor.get_vocab_size(),
        )