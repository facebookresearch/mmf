import json

import torch


class VisualDialogDatabase(torch.utils.data.Dataset):
    def __init__(self, imdb_path):
        super().__init__()
        self._load_json(imdb_path)
        self._metadata = {}

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, x):
        self._metadata = x

    def _load_json(self, imdb_path):
        with open(imdb_path, "r"):
            data = json.load(imdb_path)
            self._is_test = data["split"] == "test"
            self._question = data["questions"]
            self._answers = data["answers"]
            self._dialogs = data["dialogs"]

        # Test has only one round per dialog
        self._multiplier = 1 if self._is_test else 10
        self._qa_length = len(self._dialogs) * self._multiplier

    def __len__(self):
        return self._qa_length

    def __getitem__(self, idx):
        data = {}

        dialog_id = idx / self._multiplier
        round_id = idx % self._multiplier
        dialog = self._dialogs[dialog_id]
        data["id"] = idx
        data["dialog_id"] = dialog_id
        data["round_id"] = round_id
        round = dialog["dialog"][round_id]
        data["question"] = self._questions[round["question"]]
        # data["answers"] = [self.]
