# Copyright (c) Facebook, Inc. and its affiliates.

import random

from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset


class MaskedGQADataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            "masked_gqa", config, dataset_type, imdb_file_index, *args, **kwargs
        )
        self._add_answer = config.get("add_answer", True)

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)
            image_labels = []

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
        question = sample_info["question_str"]
        random_answer = random.choice(sample_info["all_answers"])

        processed = self.masked_token_processor(
            {"text_a": question, "text_b": random_answer, "is_correct": -1}
        )

        processed.pop("tokens")
        current_sample.update(processed)

        return current_sample
