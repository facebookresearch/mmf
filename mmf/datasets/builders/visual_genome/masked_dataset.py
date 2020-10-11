# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset


class MaskedVisualGenomeDataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            "masked_visual_genome",
            config,
            dataset_type,
            imdb_file_index,
            *args,
            **kwargs
        )
        self._add_answer = config.get("add_answer", True)

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info = self._preprocess_answer(sample_info)
        sample_info["question_id"] = sample_info["id"]
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
        if self._add_answer:
            current_sample = self.add_answer_info(sample_info, current_sample)

        return current_sample

    def _preprocess_answer(self, sample_info):
        sample_info["answers"] = [
            self.vg_answer_preprocessor(
                {"text": sample_info["answers"][0]},
                remove=["?", ",", ".", "a", "an", "the"],
            )["text"]
        ]

        return sample_info

    def add_answer_info(self, sample_info, sample):
        if "answers" in sample_info:
            answers = sample_info["answers"]
            answer_processor_arg = {"answers": answers}
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)
            sample.targets = processed_soft_copy_answers["answers_scores"]

        return sample

    def _add_masked_question(self, sample_info, current_sample):
        question = sample_info["question"]

        processed = self.masked_token_processor(
            {"text_a": question, "text_b": None, "is_correct": -1}
        )

        processed.pop("tokens")
        current_sample.update(processed)

        return current_sample
