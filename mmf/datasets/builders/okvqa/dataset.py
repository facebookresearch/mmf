# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Type, Union

import torch
from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.builders.okvqa.database import OKVQAAnnotationDatabase
from mmf.datasets.mmf_dataset import MMFDataset


class OKVQADataset(MMFDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__("okvqa", config, dataset_type, index, *args, **kwargs)

    def build_annotation_db(self) -> Type[OKVQAAnnotationDatabase]:
        annotation_path = self._get_path_based_on_index(
            self.config, "annotations", self._index
        )
        return OKVQAAnnotationDatabase(self.config, annotation_path)

    def get_image_path(self, image_id: Union[str, int]) -> str:
        if self.dataset_type == "train":
            image_path = f"COCO_train2014_{str(image_id).zfill(12)}.jpg"
        else:
            image_path = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
        return image_path

    def init_processors(self):
        super().init_processors()
        self.image_db.transform = self.image_processor

    def __getitem__(self, idx: int) -> Type[Sample]:
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
        processed_question = self.text_processor({"text": sample_info["question"]})
        current_sample.update(processed_question)
        current_sample.id = torch.tensor(
            int(sample_info["question_id"]), dtype=torch.int
        )

        # Get the first image from the set of images returned from the image_db
        image_path = self.get_image_path(sample_info["image_id"])
        current_sample.image = self.image_db.from_path(image_path)["images"][0]

        if "answers" in sample_info:
            answers = self.answer_processor({"answers": sample_info["answers"]})
            current_sample.targets = answers["answers_scores"]

        return current_sample
