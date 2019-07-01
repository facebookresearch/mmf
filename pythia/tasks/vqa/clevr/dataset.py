import os
import json

import numpy as np
import torch

from PIL import Image

from pythia.common.registry import registry
from pythia.common.sample import Sample
from pythia.tasks.base_dataset import BaseDataset
from pythia.utils.general import get_pythia_root
from pythia.utils.text_utils import VocabFromText, tokenize
from pythia.utils.distributed_utils import is_main_process, synchronize


class CLEVRDataset(BaseDataset):
    def __init__(self, dataset_type, config, data_folder=None, *args, **kwargs):
        super().__init__("clevr", dataset_type, config)
        self._data_folder = data_folder
        self._data_root_dir = os.path.join(get_pythia_root(), config.data_root_dir)

        if not self._data_folder:
            self._data_folder = os.path.join(self._data_root_dir, config.data_folder)

        if not os.path.exists(self._data_folder):
            raise RuntimeError(
                "Data folder {} for CLEVR is not present".format(self._data_folder)
            )

        # Check if the folder was actually extracted in the subfolder
        if config.data_folder in os.listdir(self._data_folder):
            self._data_folder = os.path.join(self._data_folder, config.data_folder)

        if len(os.listdir(self._data_folder)) == 0:
            raise RuntimeError("CLEVR dataset folder is empty")

        self._load()

    def _load(self):
        self.image_path = os.path.join(self._data_folder, "images", self._dataset_type)

        with open(
            os.path.join(
                self._data_folder,
                "questions",
                "CLEVR_{}_questions.json".format(self._dataset_type),
            )
        ) as f:
            self.questions = json.load(f)["questions"]

            # Only build in the main process
            if is_main_process():
                self._build_vocab(self.questions, "question")
                self._build_vocab(self.questions, "answer")
            synchronize()

    def __len__(self):
        return len(self.questions)

    def _get_vocab_path(self, attribute):
        return os.path.join(
            self._data_root_dir, "vocabs",
            "{}_{}_vocab.txt".format(self._name, attribute)
        )

    def _build_vocab(self, questions, attribute):
        # Don't build when not train
        if self._dataset_type != "train":
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
            "remove": build_attributes.get("remove", ["?", "."])
        }

        if attribute == "answer":
            kwargs["only_unk_extra"] = False

        vocab = VocabFromText(sentences, **kwargs)

        with open(vocab_file,"w") as f:
            f.write("\n".join(vocab.word_list))

    def get_item(self, idx):
        data = self.questions[idx]

        # Each call to get_item from dataloader returns a Sample class object which
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
