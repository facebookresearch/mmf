# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import json


class ImageDatabase(torch.utils.data.Dataset):
    """
    Dataset for IMDB used in Pythia
    General format that we have standardize follows:
    {
        metadata: {
            'version': x
        },
        data: [
            {
                'id': DATASET_SET_ID,
                'set_folder': <directory>,
                'feature_path': <file_path>,
                'info': {
                    // Extra information
                    'questions_tokens': [],
                    'answer_tokens': []
                }
            }
        ]
    }
    """

    def __init__(self, imdb_path):
        super().__init__()
        self.metadata = {}
        self._load_imdb(imdb_path)

    def _load_imdb(self, imdb_path):
        if imdb_path.endswith(".npy"):
            self._load_npy(imdb_path)
        elif imdb_path.endswith(".jsonl"):
            self._load_jsonl(imdb_path)
        elif imdb_path.contains("visdial") or imdb_path.contains("visual_dialog"):
            self._load_visual_dialog(imdb_path)
        else:
            raise ValueError("Unknown file format for imdb")

    def _load_jsonl(self, imdb_path):
        with open(imdb_path, "r") as f:
            db = f.readlines()
            for idx, line in enumerate(db):
                db[idx] = json.loads(line.strip("\n"))
            self.data = db
            self.start_idx = 0

    def _load_npy(self, imdb_path):
        self.db = np.load(imdb_path, allow_pickle=True)
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

    def _load_visual_dialog(self, imdb_path):
        from pythia.datasets.dialog.visual_dialog.database import VisualDialogDatabase
        self.data = VisualDialogDatabase(imdb_path)
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

        # TODO: Later clean up VizWIz IMDB from copy tokens
        if "answers" in data and data["answers"][-1] == "<copy>":
            data["answers"] = data["answers"][:-1]

        return data

    def get_version(self):
        return self.metadata.get("version", None)
