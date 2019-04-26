# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from pythia.common.sample import Sample
from pythia.tasks.vqa.vqa2 import VQA2Dataset


class VizWizDataset(VQA2Dataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(dataset_type, imdb_file_index, config, *args, **kwargs)

        # Update name as default would be 'vqa2' due to inheritance
        self._name = "vizwiz"

    def load_item(self, idx):
        sample = super().load_item(idx)

        sample_info = self.imdb[idx]

        if "image_name" in sample_info:
            sample.image_id = sample_info["image_name"]

        return sample

    def format_for_evalai(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, image_id in enumerate(report.image_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
            else:
                answer = self.answer_processor.idx2word(answer_id)
            if answer == self.context_processor.PAD_TOKEN:
                answer = "unanswerable"
            predictions.append(
                {
                    "image": "_".join(["VizWiz"] + image_id.split("_")[2:]) + ".jpg",
                    "answer": answer,
                }
            )

        return predictions
