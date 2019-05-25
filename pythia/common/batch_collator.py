# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.sample import SampleList


class BatchCollator:
    # TODO: Think more if there is a better way to do this
    _IDENTICAL_VALUE_KEYS = ["dataset_type", "dataset_name"]

    def __call__(self, batch):
        sample_list = SampleList(batch)

        for key in self._IDENTICAL_VALUE_KEYS:
            sample_list[key] = sample_list[key][0]

        return sample_list
