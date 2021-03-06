# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from mmf.common.registry import registry
from mmf.datasets.builders.retrieval.dataset import RetrievalDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("retrieval")
class RetrievalBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="retrieval", dataset_class=RetrievalDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)


@classmethod
def config_path(cls):
    return "config/datasets/retrieval/flickr30k_defaults.yaml"
