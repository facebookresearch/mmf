import json
import os

import numpy as np
import torch
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.distributed import is_main, synchronize
from mmf.utils.general import get_mmf_root
from mmf.utils.text import VocabFromText, tokenize
from PIL import Image


_CONSTANTS = {
    "questions_folder": "questions",
    "dataset_key": "clevr",
    "empty_folder_error": "CLEVR dataset folder is empty.",
    "questions_key": "questions",
    "question_key": "question",
    "answer_key": "answer",
    "train_dataset_key": "train",
    "images_folder": "images",
    "vocabs_folder": "vocabs",
}

_TEMPLATES = {
    "data_folder_missing_error": "Data folder {} for CLEVR is not present.",
    "question_json_file": "CLEVR_{}_questions.json",
    "vocab_file_template": "{}_{}_vocab.txt",
}


class CLEVRDataset(BaseDataset):
    """Dataset for CLEVR. CLEVR is a reasoning task where given an image with some
    3D shapes you have to answer basic questions.

    Args:
        dataset_type (str): type of dataset, train|val|test
        config (DictConfig): Configuration Node representing all of the data necessary
                             to initialize CLEVR dataset class
        data_folder: Root folder in which all of the data will be present if passed
                     replaces default based on data_dir and data_folder in config.

    """

    def __init__(self, config, dataset_type, data_folder=None, *args, **kwargs):
        super().__init__(_CONSTANTS["dataset_key"], config, dataset_type)
        self._data_folder = data_folder
        self._data_dir = os.path.join(get_mmf_root(), config.data_dir)

        if not self._data_folder:
            self._data_folder = os.path.join(self._data_dir, config.data_folder)

        if not os.path.exists(self._data_folder):
            raise RuntimeError(
                _TEMPLATES["data_folder_missing_error"].format(self._data_folder)
            )

        # Check if the folder was actually extracted in the subfolder
        if config.data_folder in os.listdir(self._data_folder):
            self._data_folder = os.path.join(self._data_folder, config.data_folder)

        if len(os.listdir(self._data_folder)) == 0:
            raise FileNotFoundError(_CONSTANTS["empty_folder_error"])

        self.load()

    def load(self):
        self.image_path = os.path.join(
            self._data_folder, _CONSTANTS["images_folder"], self._dataset_type
        )

        with open(
            os.path.join(
                self._data_folder,
                _CONSTANTS["questions_folder"],
                _TEMPLATES["question_json_file"].format(self._dataset_type),
            )
        ) as f:
            self.questions = json.load(f)[_CONSTANTS["questions_key"]]

            # Vocab should only be built in main process, as it will repetition of same task
            if is_main():
                self._build_vocab(self.questions, _CONSTANTS["question_key"])
                self._build_vocab(self.questions, _CONSTANTS["answer_key"])
            synchronize()

    def __len__(self):
        return len(self.questions)

    def _get_vocab_path(self, attribute):
        return os.path.join(
            self._data_dir,
            _CONSTANTS["vocabs_folder"],
            _TEMPLATES["vocab_file_template"].format(self.dataset_name, attribute),
        )

    def _build_vocab(self, questions, attribute):
        # Vocab should only be built from "train" as val and test are not observed in training
        if self._dataset_type != _CONSTANTS["train_dataset_key"]:
            return

        vocab_file = self._get_vocab_path(attribute)

        # Already exists, no need to recreate
        if os.path.exists(vocab_file):
            return

        # Create necessary dirs if not present
        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)

        sentences = [question[attribute] for question in questions]
        build_attributes = self.config.build_attributes

        # Regex is default one in tokenize i.e. space
        kwargs = {
            "min_count": build_attributes.get("min_count", 1),
            "keep": build_attributes.get("keep", [";", ","]),
            "remove": build_attributes.get("remove", ["?", "."]),
        }

        if attribute == _CONSTANTS["answer_key"]:
            kwargs["only_unk_extra"] = False

        vocab = VocabFromText(sentences, **kwargs)

        with open(vocab_file, "w") as f:
            f.write("\n".join(vocab.word_list))

    def __getitem__(self, idx):
        data = self.questions[idx]

        # Each call to __getitem__ from dataloader returns a Sample class object which
        # collated by our special batch collator to a SampleList which is basically
        # a attribute based batch in layman terms
        current_sample = Sample()

        question = data["question"]
        tokens = tokenize(question, keep=[";", ","], remove=["?", "."])
        processed = self.text_processor({"tokens": tokens})
        current_sample.text = processed["text"]

        processed = self.answer_processor({"answers": [data["answer"]]})
        current_sample.answers = processed["answers"]
        current_sample.targets = processed["answers_scores"]

        image_path = os.path.join(self.image_path, data["image_filename"])
        image = np.true_divide(Image.open(image_path).convert("RGB"), 255)
        image = image.astype(np.float32)
        current_sample.image = torch.from_numpy(image.transpose(2, 0, 1))

        return current_sample
