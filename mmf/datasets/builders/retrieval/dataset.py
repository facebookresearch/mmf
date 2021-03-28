import random

import torch
from mmf.common.sample import Sample, SampleList
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.builders.retrieval.datasets import (
    CaptionsDatabase,
    COCOAnnotationDatabase,
    ConceptualCaptionsDatabase,
)
from mmf.datasets.mmf_dataset import MMFDataset


ANNOTATIONS_DATABASE = {
    "flickr": CaptionsDatabase,
    "coco": COCOAnnotationDatabase,
    "cc": ConceptualCaptionsDatabase,
}


class RetrievalDataset(MMFDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        self.annotation_class = config.get("annotations_parser", "flickr")
        super().__init__(
            "retrieval",
            config,
            dataset_type,
            index,
            ANNOTATIONS_DATABASE[self.annotation_class],
            *args,
            **kwargs,
        )

    def init_processors(self):
        super().init_processors()
        # Assign transforms to the image_db
        if self._dataset_type == "train":
            self.image_db.transform = self.train_image_processor
        else:
            self.image_db.transform = self.eval_image_processor

    def _get_valid_text_attribute(self, sample_info):
        if "captions" in sample_info:
            return "captions"

        if "sentences" in sample_info:
            return "sentences"

        raise AttributeError("No valid text attribution was found")

    def __getitem__(self, idx):
        if self._dataset_type == "train":
            sample_info = self.annotation_db[idx]
            text_attr = self._get_valid_text_attribute(sample_info)

            current_sample = Sample()
            sentence = random.sample(sample_info[text_attr], 1)[0]
            processed_sentence = self.text_processor({"text": sentence})

            current_sample.text = processed_sentence["text"]
            if "input_ids" in processed_sentence:
                current_sample.update(processed_sentence)

            current_sample.image = self.image_db[idx]["images"][0]
            current_sample.ann_idx = torch.tensor(idx, dtype=torch.long)
        else:
            sample_info = self.annotation_db[idx]
            text_attr = self._get_valid_text_attribute(sample_info)

            sample_list = []
            for s_idx, sentence in enumerate(sample_info[text_attr]):
                sentence_sample = Sample()
                processed_sentence = self.text_processor({"text": sentence})

                sentence_sample.raw_text = sentence
                sentence_sample.text = processed_sentence["text"]
                if "input_ids" in processed_sentence:
                    sentence_sample.update(processed_sentence)

                sentence_sample.text_index = (
                    idx * self.annotation_db.samples_factor + s_idx
                )

                sample_list.append(sentence_sample)
            current_sample = SampleList(sample_list)

            current_sample.image = self.image_db[idx]["images"][0]
            current_sample.image_path = self.annotation_db[idx]["image_path"]
            current_sample.image_index = idx

        current_sample.targets = None  # Dummy for Loss

        return current_sample
