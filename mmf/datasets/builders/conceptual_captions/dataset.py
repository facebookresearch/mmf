# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from mmf.common.sample import Sample
from mmf.datasets.builders.coco import COCODataset


class ConceptualCaptionsDataset(COCODataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "conceptual_captions"

    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        processed_caption = self.text_processor({"text": sample_info["captions"][0]})
        current_sample.text = processed_caption["text"]
        current_sample.caption_len = torch.tensor(
            len(processed_caption["text"]), dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["image_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["image_id"]

        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)

        current_sample.answers = torch.stack([processed_caption["text"]])

        return current_sample
