# Copyright (c) Facebook, Inc. and its affiliates.
import json

import numpy as np
import torch
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_absolute_path


class AnnotationDatabase(torch.utils.data.Dataset):
    """
    Dataset for Annotations used in MMF

    TODO: Update on docs sprint
    """

    def __init__(self, config, path, *args, **kwargs):
        super().__init__()
        self.metadata = {}
        self.config = config
        self.start_idx = 0
        path = get_absolute_path(path)
        self.load_annotation_db(path)

    def load_annotation_db(self, path):
        if path.find("visdial") != -1 or path.find("visual_dialog") != -1:
            self._load_visual_dialog(path)
        elif path.endswith(".npy"):
            self._load_npy(path)
        elif path.endswith(".jsonl"):
            self._load_jsonl(path)
        elif path.endswith(".json"):
            self._load_json(path)
        else:
            raise ValueError("Unknown file format for annotation db")

    def _load_jsonl(self, path):
        with PathManager.open(path, "r") as f:
            db = f.readlines()
            for idx, line in enumerate(db):
                db[idx] = json.loads(line.strip("\n"))
            self.data = db
            self.start_idx = 0

    def _load_npy(self, path):
        with PathManager.open(path, "rb") as f:
            self.db = np.load(f, allow_pickle=True)

        self.start_idx = 0

        if type(self.db) == dict:
            self.metadata = self.db.get("metadata", {})
            self.data = self.db.get("data", [])
        else:
            # TODO: Deprecate support for this
            self.metadata = {"version": 1}
            self.data = self.db
            # Handle old imdb support
            if "image_id" not in self.data[0]:
                self.start_idx = 1

        if len(self.data) == 0:
            self.data = self.db

    def _load_json(self, path):
        with PathManager.open(path, "r") as f:
            data = json.load(f)
        self.metadata = data.get("metadata", {})
        self.data = data.get("data", [])

        if len(self.data) == 0:
            raise RuntimeError("Dataset is empty")

    def _load_visual_dialog(self, path):
        from mmf.datasets.builders.visual_dialog.database import VisualDialogDatabase

        self.data = VisualDialogDatabase(path)
        self.metadata = self.data.metadata
        self.start_idx = 0

    def __len__(self):
        return len(self.data) - self.start_idx

    def __getitem__(self, idx):
        data = self.data[idx + self.start_idx]

        # Hacks for older IMDBs
        if "answers" not in data:
            if "all_answers" in data and "valid_answers" not in data:
                data["answers"] = data["all_answers"]
            if "valid_answers" in data:
                data["answers"] = data["valid_answers"]

        # TODO: Clean up VizWiz IMDB from copy tokens
        if "answers" in data and data["answers"][-1] == "<copy>":
            data["answers"] = data["answers"][:-1]

        return data

    def get_version(self):
        return self.metadata.get("version", None)
