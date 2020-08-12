# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import shutil

from mmf.common.constants import VISUAL_DIALOG_CONSTS
from mmf.common.registry import registry
from mmf.datasets.builders.visual_dialog.dataset import VisualDialogDataset
from mmf.datasets.builders.visual_genome.builder import VisualGenomeBuilder
from mmf.utils.download import decompress, download
from mmf.utils.general import get_mmf_root


logger = logging.getLogger(__name__)


@registry.register_builder("visual_dialog")
class VisualDialogBuilder(VisualGenomeBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "visual_dialog"
        self.dataset_class = VisualDialogDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/visual_dialog/defaults.yaml"

    def build(self, config, dataset_type):
        self._dataset_type = dataset_type
        self._config = config
        data_folder = os.path.join(get_mmf_root(), self._config.data_dir)

        self._download_and_extract_imdb(data_folder)

        if self._dataset_type != "train":
            return

        self._download_and_extract(
            "vocabs", VISUAL_DIALOG_CONSTS["vocabs"], data_folder
        )
        self._download_and_extract_features(data_folder)

    def _download_and_extract_imdb(self, data_folder):
        download_folder = os.path.join(data_folder, "imdb")

        self._download_and_extract(
            "imdb_url",
            VISUAL_DIALOG_CONSTS["imdb_url"][self._dataset_type],
            download_folder,
        )

    def _download_and_extract_features(self, data_folder):
        # Visual Dialog features will contain val and test
        self._download_and_extract(
            "features_url",
            VISUAL_DIALOG_CONSTS["features_url"]["visual_dialog"],
            data_folder,
        )
        # But since train is same as COCO, we reuse those features if already downloaded
        self._download_and_extract(
            "features_url", VISUAL_DIALOG_CONSTS["features_url"]["coco"], data_folder
        )
