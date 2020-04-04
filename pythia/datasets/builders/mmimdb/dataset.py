import copy
import json

import torch

from pythia.common.sample import Sample
from pythia.datasets.builders.vqa2 import VQA2Dataset


class MMIMDbDataset(VQA2Dataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(dataset_type, imdb_file_index, config, *args, **kwargs)

        self._name = "mmimdb"

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        current_sample = Sample()
        plot = sample_info["plot"]
        if isinstance(plot, list):
            plot = plot[0]
        processed_sentence = self.text_processor({"text": plot})

        current_sample.text = processed_sentence["text"]
        if "input_ids" in processed_sentence:
            current_sample.update(processed_sentence)

        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)

        processed = self.answer_processor({"answers": sample_info["genres"]})
        current_sample.answers = processed["answers"]
        current_sample.targets = processed["answers_scores"]

        return current_sample
