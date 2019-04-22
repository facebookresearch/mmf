# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.tasks.vqa.vizwiz import VizWizDataset
from pythia.utils.text_utils import word_tokenize


class VQA2OCRDataset(VizWizDataset):
    def __init__(self, imdb_file, image_feat_directories, verbose=False, **data_params):
        super(VQA2OCRDataset, self).__init__(
            imdb_file, image_feat_directories, verbose, **data_params
        )
        self.name = "vqa2_ocr"

    def format_for_evalai(self, batch, answers):
        answers = answers.argmax(dim=1)

        predictions = []
        for idx, question_id in enumerate(batch["question_id"]):
            answer_id = answers[idx]

            if answer_id >= self.answer_space_size:
                answer_id -= self.answer_space_size
                answer = word_tokenize(batch["ocr_tokens"][answer_id][idx])
            else:
                answer = self.answer_dict.idx2word(answer_id)
            predictions.append({"question_id": question_id.item(), "answer": answer})

        return predictions

    def __getitem__(self, idx):
        sample = super(VQA2OCRDataset, self).__getitem__(idx)

        if sample["question_id"] is None:
            sample["question_id"] = -1
        return sample
