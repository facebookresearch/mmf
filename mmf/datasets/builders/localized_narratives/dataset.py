# Copyright (c) Facebook, Inc. and its affiliates.

import time

from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.builders.localized_narratives.database import (
    LocalizedNarrativesAnnotationDatabase,
)
from mmf.datasets.mmf_dataset import MMFDataset


class LocalizedNarrativesDataset(MMFDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "localized_narratives", config, dataset_type, index, *args, **kwargs
        )

    def build_annotation_db(self) -> LocalizedNarrativesAnnotationDatabase:
        annotation_path = self._get_path_based_on_index(
            self.config, "annotations", self._index
        )
        return LocalizedNarrativesAnnotationDatabase(self.config, annotation_path)

    def __getitem__(self, idx: int) -> Sample:
        start_time = time.time()
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
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

        image_info = current_sample["image_info_0"]
        if hasattr(self, "transformer_bbox_processor"):
            image_info = self.transformer_bbox_processor(image_info)

        processed = self.text_image_processor(
            {
                "utterances": sample_info["utterances"],
                "utterance_times": sample_info["utterance_times"],
                "image_info": image_info,
                "traces": sample_info["traces"],
            }
        )
        current_sample.update(processed)
        print(
            "Duration: ",
            time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
        )
        return current_sample
