# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.sample import SampleList


class BatchCollator:
    def __init__(self, dataset_name, dataset_type):
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type

    def __call__(self, batch):
        # Create and return sample list with proper name
        # and type set if it is already not a sample list
        # (case of batched iterators)
        sample_list = batch
        if (
            # Check if batch is a list before checking batch[0]
            # or len as sometimes batch is already SampleList
            isinstance(batch, list)
            and len(batch) == 1
            and isinstance(batch[0], SampleList)
        ):
            sample_list = batch[0]
        elif not isinstance(batch, SampleList):
            sample_list = SampleList(batch)

        if sample_list._get_tensor_field() is None:
            sample_list = SampleList(sample_list.to_dict())

        sample_list.dataset_name = self._dataset_name
        sample_list.dataset_type = self._dataset_type
        return sample_list
