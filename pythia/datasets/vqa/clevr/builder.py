import json
import math
import os
import zipfile
from collections import Counter

from pythia.common.registry import registry
from pythia.common.constants import CLEVR_DOWNLOAD_URL
from pythia.datasets.base_dataset_builder import BaseDatasetBuilder
from pythia.datasets.vqa.clevr.dataset import CLEVRDataset
from pythia.utils.general import download_file, get_pythia_root


@registry.register_builder("clevr")
class CLEVRBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("clevr")
        self.writer = registry.get("writer")
        self.dataset_class = CLEVRDataset

    def _build(self, dataset_type, config):
        download_folder = os.path.join(get_pythia_root(), config.data_root_dir, config.data_folder)

        file_name = CLEVR_DOWNLOAD_URL.split("/")[-1]
        local_filename = os.path.join(download_folder, file_name)

        extraction_folder = os.path.join(download_folder, ".".join(file_name.split(".")[:-1]))
        self.data_folder = extraction_folder

        # Either if the zip file is already present or if there are some
        # files inside the folder we don't continue download process
        if os.path.exists(local_filename):
            self.writer.write("CLEVR dataset is already present. Skipping download.")
            return

        if os.path.exists(extraction_folder) and \
            len(os.listdir(extraction_folder)) != 0:
            return

        self.writer.write("Downloading the CLEVR dataset now")
        download_file(CLEVR_DOWNLOAD_URL, output_dir=download_folder)

        self.writer.write("Downloaded. Extracting now. This can take time.")
        with zipfile.ZipFile(local_filename, "r") as zip_ref:
            zip_ref.extractall(download_folder)


    def _load(self, dataset_type, config, *args, **kwargs):
        self.dataset = CLEVRDataset(
            dataset_type, config, data_folder=self.data_folder
        )
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
