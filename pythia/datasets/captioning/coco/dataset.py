# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from pythia.common.sample import Sample
from pythia.datasets.vqa.vqa2 import VQA2Dataset


class COCODataset(VQA2Dataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(dataset_type, imdb_file_index, config, *args, **kwargs)
        self._name = "coco"

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        current_sample = Sample()

        if self._dataset_type != "test":
            text_processor_argument = {"tokens": sample_info["caption_tokens"]}
            processed_caption = self.text_processor(text_processor_argument)
            current_sample.text = processed_caption["text"]
            current_sample.caption_id = torch.tensor(
                sample_info["caption_id"], dtype=torch.int
            )
            current_sample.caption_len = torch.tensor(
                len(sample_info["caption_tokens"]), dtype=torch.int
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

        # Add reference captions to sample
        current_sample = self.add_reference_caption(sample_info, current_sample)

        return current_sample

    def add_reference_caption(self, sample_info, sample):
        reference_list = []
        for reference in sample_info["reference_tokens"]:
            text_processor_argument = {"tokens": reference}
            processed_reference = self.text_processor(text_processor_argument)
            reference_list.append(processed_reference["text"])

        # Restrict to minimum reference captions available per image
        sample.answers = torch.stack(reference_list)[: self.config.min_captions_per_img]

        return sample

    def format_for_evalai(self, report):
        captions = report.captions.tolist()
        predictions = []
        remove_unk_from_caption_prediction = getattr(
            self.config, 'remove_unk_from_caption_prediction', False
        )
        for idx, image_id in enumerate(report.image_id):
            caption = self.caption_processor(captions[idx])["caption"]
            if remove_unk_from_caption_prediction:
                caption = caption.replace('<unk>', '')
                caption = caption.replace('  ', ' ').strip()
            if isinstance(image_id, torch.Tensor):
                image_id = image_id.item()
            predictions.append({"image_id": image_id, "caption": caption})

        return predictions
