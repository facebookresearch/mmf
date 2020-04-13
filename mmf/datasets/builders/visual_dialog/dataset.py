import copy
import json

import torch

from mmf.common.sample import Sample
from mmf.datasets.builders.visual_dialog.database import VisualDialogDatabase
from mmf.datasets.builders.vqa2 import VQA2Dataset


class VisualDialogDataset(VQA2Dataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(dataset_type, imdb_file_index, config, *args, **kwargs)

        self._name = "visual_dialog"

        discriminative = config.discriminative
        self._discriminative = discriminative.enabled
        self._return_indices = discriminative.return_indices
        self._no_unk = config.no_unk
        self._return_history = config.return_history
