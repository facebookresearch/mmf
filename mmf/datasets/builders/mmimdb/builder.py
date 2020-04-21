# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.mmimdb.dataset import MMIMDbDataset
from mmf.datasets.builders.vqa2.builder import VQA2Builder
from mmf.utils.download import DownloadableFile

IMAGE_RESOURCES = DownloadableFile(
    "https://archive.org/download/mmimdb/mmimdb.tar.gz",
    "mmimdb.tar.gz",
    hashcode="7facb412f84e8e707cf5c15bb58e4cf3ac12d33e6e944e1bdefebada1259a253",
)

FEATURE_RESOURCES = DownloadableFile(
    "https://dl.fbaipublicfiles.com/mmf/data/datasets/mmimdb/"
    + "features/features.lmdb.tar.gz",
    "features.lmdb.tar.gz",
    hashcode="dab8ef859d872fa42f84eb6f710d408929bc35f056ca699b47e8a43d0657f3f1",
)

ANNOTATION_RESOURCES = DownloadableFile(
    "https://dl.fbaipublicfiles.com/mmf/data/datasets/mmimdb/"
    + "annotations/mmimdb.tar.gz",
    "mmimdb.tar.gz",
    hashcode="5df7486bf143b073b7fb8a1738ce806876529d30a8aa13b967614d8a6c72a360",
)

EXTRA_RESOURCES = DownloadableFile(
    "https://dl.fbaipublicfiles.com/mmf/data/datasets/mmimdb/extras.tar.gz",
    "extras.tar.gz",
    hashcode="08dd544c152c54ca37330f77d4ed40a29ab0646afea855568e9c0e6ffd86b935",
)


@registry.register_builder("mmimdb")
class MMIMDbBuilder(VQA2Builder):
    VERSION = "1.0_2020_04_16"
    RESOURCES = {
        "images": IMAGE_RESOURCES,
        "features": FEATURE_RESOURCES,
        "annotations": ANNOTATION_RESOURCES,
        "extras": EXTRA_RESOURCES,
    }

    def __init__(self):
        super().__init__()
        self.dataset_name = "mmimdb"
        self.dataset_class = MMIMDbDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/mmimdb/defaults.yaml"
