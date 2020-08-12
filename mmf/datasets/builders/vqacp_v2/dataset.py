# Copyright (c) Facebook, Inc. and its affiliates.
import os
from typing import Type, Union

import torch
from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.builders.okvqa.dataset import OKVQADataset
from mmf.datasets.builders.vqacp_v2.database import VQACPv2AnnotationDatabase


class VQACPv2Dataset(OKVQADataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(config, dataset_type, index, *args, **kwargs)

    def build_annotation_db(self) -> Type[VQACPv2AnnotationDatabase]:
        annotation_path = self._get_path_based_on_index(
            self.config, "annotations", self._index
        )
        return VQACPv2AnnotationDatabase(self.config, annotation_path)

    def get_image_path(self, image_id: Union[str, int], coco_split: str) -> str:
        base_paths = self._get_path_based_on_index(self.config, "images", self._index)
        base_paths = base_paths.split(",")
        if "train" in base_paths[0]:
            train_path = base_paths[0]
            val_path = base_paths[1]
        else:
            train_path = base_paths[1]
            val_path = base_paths[0]

        # coco_split indicates whether the image is from the train or val split of COCO
        if "train" in coco_split:
            image_path = f"COCO_train2014_{str(image_id).zfill(12)}.jpg"
            image_path = os.path.join(train_path, image_path)
        else:
            image_path = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
            image_path = os.path.join(val_path, image_path)
        return image_path

    def __getitem__(self, idx: int) -> Type[Sample]:
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
        processed_question = self.text_processor({"text": sample_info["question"]})
        current_sample.update(processed_question)
        current_sample.id = torch.tensor(
            int(sample_info["question_id"]), dtype=torch.int
        )

        image_path = self.get_image_path(
            sample_info["image_id"], sample_info["coco_split"]
        )
        current_sample.image = self.image_db.from_path(image_path)["images"][0]

        if "answers" in sample_info:
            answers = self.answer_processor({"answers": sample_info["answers"]})
            current_sample.targets = answers["answers_scores"]

        return current_sample
