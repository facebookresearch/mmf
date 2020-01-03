# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.datasets.vqa.m4c_textvqa.dataset import M4CTextVQADataset


class M4CSTVQADataset(M4CTextVQADataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(
            dataset_type, imdb_file_index, config, *args, **kwargs
        )
        self._name = "m4c_stvqa"
