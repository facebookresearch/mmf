# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.datasets.builders.m4c_textvqa.dataset import M4CTextVQADataset


class STVQADataset(M4CTextVQADataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "stvqa"
