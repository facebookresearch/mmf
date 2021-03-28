# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.sample import convert_batch_to_sample_list


class BatchCollator:
    def __init__(self, dataset_name, dataset_type):
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type

    def __call__(self, batch):
        sample_list = convert_batch_to_sample_list(batch)
        sample_list.dataset_name = self._dataset_name
        sample_list.dataset_type = self._dataset_type
        return sample_list
