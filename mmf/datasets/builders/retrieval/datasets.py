import json

import pandas as pd
import torch
from mmf.utils.file_io import PathManager


class CaptionsDatabase(torch.utils.data.Dataset):
    """
    Dataset for Flickr Annotations
    """

    SPLITS = {"train": ["train"], "val": ["val"], "test": ["test"]}

    def __init__(self, config, splits_path, dataset_type: str, *args, **kwargs):
        super().__init__()
        self.config = config
        self.dataset_type = dataset_type
        self.splits = self.SPLITS[self.dataset_type]
        self._load_annotation_db(splits_path)

    def _load_annotation_db(self, splits_path):
        data = []

        with PathManager.open(splits_path, "r") as f:
            annotations_json = json.load(f)

        for image in annotations_json["images"]:
            if image["split"] in self.splits:
                data.append(
                    {
                        "image_path": image["filename"],
                        "sentences": [s["raw"] for s in image["sentences"]],
                    }
                )

        if len(data) == 0:
            raise RuntimeError("Dataset is empty")

        self.samples_factor = len(data[0]["sentences"])
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class COCOAnnotationDatabase(CaptionsDatabase):
    """
    Dataset for COCO Annotations with extra 30K samples for training
    """

    SPLITS = {"train": ["train", "restval"], "val": ["val"], "test": ["test"]}

    def _load_annotation_db(self, splits_path):
        data = []

        with PathManager.open(splits_path, "r") as f:
            annotations_json = json.load(f)

        for image in annotations_json["images"]:
            if image["split"] in self.splits:
                image_path = image["filename"]
                # hard-fix for the extra images from val
                if image["split"] == "train":
                    image_path = "../train2014/" + image_path
                elif image["split"] == "restval":
                    image_path = "../val2014/" + image_path
                elif image["split"] == "val":
                    image_path = "../val2014/" + image_path
                elif image["split"] == "test":
                    image_path = "../val2014/" + image_path
                else:
                    raise NotImplementedError

                data.append(
                    {
                        "image_path": image_path,
                        # Cap for 5 captions
                        "sentences": [s["raw"] for s in image["sentences"][:5]],
                    }
                )

        if len(data) == 0:
            raise RuntimeError("Dataset is empty")

        self.samples_factor = len(data[0]["sentences"])
        self.data = data


class ConceptualCaptionsDatabase(CaptionsDatabase):
    """
    Dataset for conceptual caption database
    """

    SPLITS = {"train": ["train"], "val": ["val"], "test": ["test"]}

    def _load_annotation_db(self, splits_path):
        df = pd.read_csv(
            splits_path, compression="gzip", sep="\t", names=["caption", "file"]
        )

        self.data = df

        self.samples_factor = 1

        if len(self.data) == 0:
            raise RuntimeError("Dataset is empty")

    def __getitem__(self, idx):
        df_i = self.data.iloc[idx]
        data = {"sentences": [df_i["caption"]], "image_path": df_i["file"]}
        return data
