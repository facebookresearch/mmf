# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import random

from mmf.datasets.mmf_dataset import MMFDataset


logger = logging.getLogger(__name__)


class VinVLDataset(MMFDataset):
    """The VinVL dataset is a dataset that augments an existing
    dataset within MMF. VinVL requires unique inputs for
    finetuning and pretraining unsupported by general datasets.
    To enable this functionality on arbitrary datasets,
    the VinVL dataset contains a base dataset,
    and returns an augmented version of samples from the
    base dataset.

    For example, the VQA2 dataset may return a sample {image, text}
    The VinVL dataset when asked for a sample, will return
    {image, text', rand_caption, rand_label}
        text' = text + labels
        rand_caption = text from a random example
        rand_label = obj detection labels text for a random example

    Why does this exist?
    VinVL samples contain rand_caption, and rand_label which require
    random choice from the annotations db, and features_db.
    Currently general text_processors do not have access to these
    databases, instead randomness like mismatched_captions in
    masked coco are implemented on the dataset level.
    To support VinVL finetuning and pretraining on general datasets,
    without a major refactor, the VinVL builder and dataset introduce
    a new design pattern to enable processor access to databases.

    Interface and Assumptions:
    The VinVL dataset assumes:
    The sample returned by the base dataset contains a key "text"
    with string text.
    There exists a label_map json file path in the dataset config
    for a json obj containing idx_to_attribute and idx_to_label
    maps. VinVL OD uses VG labels, and this map can be downloaded
    from https://penzhanwu2.blob.core.windows.net/sgg/
    sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json
    The features_db points to features generated from the VinVL
    feature extraction script, consult the VinVL feature
    extraction tutorial for more details.
    """

    def __init__(self, config, dataset_type, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "vinvl"
        super().__init__(name, config, dataset_type, *args, **kwargs)
        self.add_tags = not "test" == self._dataset_type
        self.label_map = self.load_label_map(config.get("label_map"))

    def set_base_dataset(self, base_dataset):
        self.base_dataset = base_dataset

    def init_processors(self):
        super().init_processors()

    def __len__(self):
        return len(self.annotation_db)

    def __getitem__(self, idx):
        return self.load_item(idx)

    def load_item(self, idx):
        base_sample = self.base_dataset.load_item(idx)
        # assumes sample contains key "text" that is the string text
        # when using on vqa2 which returns tokens under key "text"
        # change the vqa2 dataset class to return "text"
        text_processor_argument = {"text": base_sample["text"]}
        if self.add_tags:
            text_processor_argument["text_b"] = self.get_label_str(base_sample)

            random_caption_idx = random.randint(0, len(self.annotation_db) - 1)
            random_caption_sample = self.base_dataset.load_item(random_caption_idx)
            random_caption = random_caption_sample["text"]
            text_processor_argument["random_captions"] = [random_caption]

            random_labels_idx = random.randint(0, len(self.annotation_db) - 1)
            random_labels_sample = self.base_dataset.load_item(random_labels_idx)
            random_image_tags_str = self.get_label_str(random_labels_sample)
            text_processor_argument["random_labels"] = [random_image_tags_str]

        processed_caption = self.text_processor(text_processor_argument)
        base_sample.update(processed_caption)
        return base_sample

    def load_label_map(self, map_path):
        with open(map_path) as f:
            return json.loads(f.read())

    def get_label_str(self, sample):
        image_labels = sample["image_info_0"].get("labels", [])
        label_map = self.label_map.get("idx_to_label", {})
        label_str = " ".join([label_map.get(str(id), "") for id in image_labels])
        image_attr_labels = sample["image_info_0"].get("attr_labels", [])
        attr_map = self.label_map.get("idx_to_attribute", {})
        attr_str = " ".join([attr_map.get(str(id), "") for id in image_attr_labels])
        accum_str = label_str + " " + attr_str
        return accum_str
