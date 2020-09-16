# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Type, Union

import torch
from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.builders.localized_narratives.database import (
    LocalizedNarrativesAnnotationDatabase,
)
from mmf.datasets.mmf_dataset import MMFDataset


class MaskedLocalizedNarrativesDataset(MMFDataset):
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

    def build_annotation_db(self) -> Type[LocalizedNarrativesAnnotationDatabase]:
        annotation_path = self._get_path_based_on_index(
            self.config, "annotations", self._index
        )
        return LocalizedNarrativesAnnotationDatabase(self.config, annotation_path)

    def __getitem__(self, idx: int) -> Type[Sample]:
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
        processed_caption = self.masked_token_processor(
            {"text_a": sample_info["caption"], "text_b": "", "is_correct": True,}
        )
        current_sample.update(processed_caption)

        # Get the image features
        if self._use_features:
            features = self.features_db[idx]
            current_sample.update(features)

        return current_sample
