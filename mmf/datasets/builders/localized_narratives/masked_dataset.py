# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABC

from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.builders.localized_narratives.database import (
    LocalizedNarrativesAnnotationDatabase,
)
from mmf.datasets.mmf_dataset import MMFDataset


class MaskedLocalizedNarrativesDatasetMixin(ABC):
    def build_annotation_db(self) -> LocalizedNarrativesAnnotationDatabase:
        annotation_path = self._get_path_based_on_index(
            self.config, "annotations", self._index
        )
        return LocalizedNarrativesAnnotationDatabase(self.config, annotation_path)

    def __getitem__(self, idx: int) -> Sample:
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
        processed_caption = self.masked_token_processor(
            {"text_a": sample_info["caption"], "text_b": "", "is_correct": True}
        )
        current_sample.update(processed_caption)
        current_sample.image_id = sample_info["image_id"]
        current_sample.feature_path = sample_info["feature_path"]

        # Get the image features
        if self._use_features:
            features = self.features_db[idx]
            image_info_0 = features["image_info_0"]
            if image_info_0 and "image_id" in image_info_0.keys():
                image_info_0["feature_path"] = image_info_0["image_id"]
                image_info_0.pop("image_id")
            current_sample.update(features)

        return current_sample


class MaskedLocalizedNarrativesDataset(
    MaskedLocalizedNarrativesDatasetMixin, MMFDataset
):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "masked_localized_narratives", config, dataset_type, index, *args, **kwargs
        )
