import json
import logging
import math
import os
import zipfile
from collections import Counter

from mmf.common.constants import CLEVR_DOWNLOAD_URL
from mmf.common.registry import registry
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.datasets.builders.clevr.dataset import CLEVRDataset
from mmf.utils.download import download
from mmf.utils.general import get_mmf_root


logger = logging.getLogger(__name__)


@registry.register_builder("clevr")
class CLEVRBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("clevr")
        self.dataset_class = CLEVRDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/clevr/defaults.yaml"

    def build(self, config, dataset_type):
        download_folder = os.path.join(
            get_mmf_root(), config.data_dir, config.data_folder
        )

        file_name = CLEVR_DOWNLOAD_URL.split("/")[-1]
        local_filename = os.path.join(download_folder, file_name)

        extraction_folder = os.path.join(
            download_folder, ".".join(file_name.split(".")[:-1])
        )
        self.data_folder = extraction_folder

        # Either if the zip file is already present or if there are some
        # files inside the folder we don't continue download process
        if os.path.exists(local_filename):
            logger.info("CLEVR dataset is already present. Skipping download.")
            return

        if (
            os.path.exists(extraction_folder)
            and len(os.listdir(extraction_folder)) != 0
        ):
            return

        logger.info("Downloading the CLEVR dataset now")
        download(CLEVR_DOWNLOAD_URL, download_folder, CLEVR_DOWNLOAD_URL.split("/")[-1])

        logger.info("Downloaded. Extracting now. This can take time.")
        with zipfile.ZipFile(local_filename, "r") as zip_ref:
            zip_ref.extractall(download_folder)

    def load(self, config, dataset_type, *args, **kwargs):
        self.dataset = CLEVRDataset(config, dataset_type, data_folder=self.data_folder)
        return self.dataset

    def update_registry_for_model(self, config):
        registry.register(
            self.dataset_name + "_text_vocab_size",
            self.dataset.text_processor.get_vocab_size(),
        )
        registry.register(
            self.dataset_name + "_num_final_outputs",
            self.dataset.answer_processor.get_vocab_size(),
        )
