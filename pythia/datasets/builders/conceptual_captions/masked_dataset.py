# Copyright (c) Facebook, Inc. and its affiliates.

from pythia.datasets.builders.coco import MaskedCOCODataset


class MaskedConceptualCaptionsDataset(MaskedCOCODataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(dataset_type, imdb_file_index, config, *args, **kwargs)
        self._name = "masked_conceptual_captions"
        self._two_sentence = config.get("two_sentence", True)
        self._false_caption = config.get("false_caption", True)
        self._two_sentence_probability = config.get("two_sentence_probability", 0.5)
        self._false_caption_probability = config.get("false_caption_probability", 0.5)
