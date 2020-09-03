import random

from mmf.common.sample import Sample
from mmf.datasets.builders.vqa2.dataset import VQA2Dataset


class MaskedVQA2Dataset(VQA2Dataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            config,
            dataset_type,
            imdb_file_index,
            dataset_name="masked_vqa2",
            *args,
            **kwargs
        )
        self._add_answer = config.get("add_answer", False)

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        if self._use_features:
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
        else:
            image_path = str(sample_info["image_name"]) + ".jpg"
            current_sample.image = self.image_db.from_path(image_path)["images"][0]

        current_sample = self._add_masked_question(sample_info, current_sample)
        if self._add_answer:
            current_sample = self.add_answer_info(sample_info, current_sample)
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
