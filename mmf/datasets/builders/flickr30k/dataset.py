# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np

from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset


class Flickr30kDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="visual7w", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)

    def __getitem__(self, idx):

        sample_info = self.annotation_db[idx]
        image_id = str(sample_info[0]['image_id']) + '.npy'
        features_data = self.features_db.from_path(image_id)

        current_sample = Sample()

        return current_sample
