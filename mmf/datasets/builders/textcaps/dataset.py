# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.datasets.builders.textvqa.dataset import TextVQADataset
from mmf.utils.distributed import object_to_byte_tensor


class TextCapsDataset(TextVQADataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "textcaps"

    def preprocess_sample_info(self, sample_info):
        sample_info = super().preprocess_sample_info(sample_info)
        # add dummy questions to train with M4C (for TextVQA)
        sample_info["question_str"] = ""  # empty question
        sample_info["question_id"] = sample_info["caption_id"]
        return sample_info

    def postprocess_evalai_entry(self, entry):
        new_entry = {
            "caption_id": entry["question_id"],
            "image_id": entry["image_id"],
            "caption": entry["answer"],
            "pred_source": entry["pred_source"],
        }
        return new_entry

    def add_answer_info(self, sample_info, sample):
        sample_has_caption = "caption_str" in sample_info
        if sample_has_caption:
            sample_info["answers"] = [sample_info["caption_str"]]

        sample = super().add_answer_info(sample_info, sample)

        if sample_has_caption:
            sample.caption_str = object_to_byte_tensor(sample_info["caption_str"])
            sample.ref_strs = object_to_byte_tensor(sample_info["reference_strs"])
            sample.pop("answers")

        return sample
