import random

from mmf.common.sample import Sample
from mmf.datasets.builders.coco import COCODataset


class MaskedCOCODataset(COCODataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "masked_coco"
        self._two_sentence = config.get("two_sentence", True)
        self._false_caption = config.get("false_caption", True)
        self._two_sentence_probability = config.get("two_sentence_probability", 0.5)
        self._false_caption_probability = config.get("false_caption_probability", 0.5)

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        current_sample = Sample()

        if self._use_features is True:
            features = self.features_db[idx]
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
            current_sample.update(features)

        current_sample = self._add_masked_caption(sample_info, current_sample)
        return current_sample

    def _add_masked_caption(self, sample_info, current_sample):
        captions = sample_info["captions"]
        image_id = sample_info["image_id"]
        num_captions = len(captions)
        selected_caption_index = random.randint(0, num_captions - 1)
        other_caption_indices = [
            i for i in range(num_captions) if i != selected_caption_index
        ]
        selected_caption = captions[selected_caption_index]
        other_caption = None
        is_correct = -1

        if self._dataset_type == "train":
            if self._two_sentence:
                if random.random() > self._two_sentence_probability:
                    other_caption = self._get_mismatching_caption(image_id)
                    is_correct = False
                else:
                    other_caption = captions[random.choice(other_caption_indices)]
                    is_correct = True
            elif self._false_caption:
                if random.random() < self._false_caption_probability:
                    selected_caption = self._get_mismatching_caption(image_id)
                    is_correct = False
                else:
                    is_correct = True

        processed = self.masked_token_processor(
            {
                "text_a": selected_caption,
                "text_b": other_caption,
                "is_correct": is_correct,
            }
        )
        processed.pop("tokens")
        current_sample.update(processed)

        return current_sample

    def _get_mismatching_caption(self, image_id):
        other_item = self.imdb[random.randint(0, len(self.imdb) - 1)]

        while other_item["image_id"] == image_id:
            other_item = self.imdb[random.randint(0, len(self.imdb) - 1)]

        other_caption = other_item["captions"][
            random.randint(0, len(other_item["captions"]) - 1)
        ]
        return other_caption
