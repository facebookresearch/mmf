# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from mmf.common.sample import Sample
from mmf.datasets.builders.vqa2 import VQA2Dataset
from mmf.utils.distributed import byte_tensor_to_object, object_to_byte_tensor


class COCODataset(VQA2Dataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            config, dataset_type, imdb_file_index, dataset_name="coco", *args, **kwargs
        )

    def preprocess_sample_info(self, sample_info):
        # COCO Annotation DBs have corrext feature_path
        if "COCO" not in sample_info["feature_path"]:
            sample_info["feature_path"] = sample_info["image_path"].replace(
                ".jpg", ".npy"
            )
        return sample_info

    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info = self.preprocess_sample_info(sample_info)
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

        current_sample.image_id = object_to_byte_tensor(sample_info["image_id"])

        if self._use_features:
            features = self.features_db[idx]
            current_sample.update(features)
        else:
            image_path = str(sample_info["image_name"]) + ".jpg"
            current_sample.image = self.image_db.from_path(image_path)["images"][0]

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

    def format_for_prediction(self, report):
        captions = report.captions.tolist()
        predictions = []
        remove_unk_from_caption_prediction = getattr(
            self.config, "remove_unk_from_caption_prediction", False
        )

        for idx, image_id in enumerate(report.image_id):
            image_id = byte_tensor_to_object(image_id)
            caption = self.caption_processor(captions[idx])["caption"]
            if remove_unk_from_caption_prediction:
                caption = caption.replace("<unk>", "")
                caption = caption.replace("  ", " ").strip()
            if isinstance(image_id, torch.Tensor):
                image_id = image_id.item()
            predictions.append({"image_id": image_id, "caption": caption})

        return predictions
