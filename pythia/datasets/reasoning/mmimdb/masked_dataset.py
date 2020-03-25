import random

import numpy as np

from pythia.common.sample import Sample
from pythia.datasets.vqa.vqa2.dataset import VQA2Dataset


class MaskedMMImdbDataset(VQA2Dataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(dataset_type, imdb_file_index, config, *args, **kwargs)
        self._name = "masked_mmimdb"
        self._add_answer = config.get("add_answer", True)

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        current_sample = Sample()

        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)
            image_labels = []
            overlaps = np.ones((features["image_feature_0"].shape[0]))

            for i in range(features["image_feature_0"].shape[0]):
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    if prob < 0.9:
                        features["image_feature_0"][i] = 0
                    image_labels.append(1)
                else:
                    # no masking token (will be ignored by loss function later)
                    image_labels.append(-1)
            item = {}
            if self.config.get("use_image_feature_masks", False):
                item["image_labels"] = image_labels
            current_sample.update(item)
        current_sample = self._add_masked_question(sample_info, current_sample)

        return current_sample

    def _add_masked_question(self, sample_info, current_sample):
        plot = sample_info["plot"]
        if isinstance(plot, list):
            plot = plot[0]
        question = plot
        random_answer = random.choice(sample_info["genres"])

        processed = self.masked_token_processor(
            {"text_a": question, "text_b": random_answer, "is_correct": -1}
        )

        processed.pop("tokens")
        current_sample.update(processed)

        return current_sample
