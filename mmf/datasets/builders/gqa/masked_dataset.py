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

            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )

            if self.config.get("use_image_feature_masks", False):
                current_sample.update(
                    {
                        "image_labels": self.masked_region_processor(
                            features["image_feature_0"]
                        )
                    }
                )

            current_sample.update(features)

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
