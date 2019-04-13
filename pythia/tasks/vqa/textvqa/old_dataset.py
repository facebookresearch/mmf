import torch

from collections import Counter
from pythia.utils.text_utils import word_tokenize

from pythia.tasks.vqa.vizwiz import VizWizDataset


class TextVQADataset(VizWizDataset):
    def __init__(self, imdb_file, image_feat_directories, verbose=False,
                 **data_params):
        super(TextVQADataset, self).__init__(imdb_file, image_feat_directories,
                                             verbose, **data_params)

        self.name = "textvqa"

    def format_for_evalai(self, batch, answers):
        answers = answers.argmax(dim=1)

        predictions = []
        for idx, question_id in enumerate(batch['question_id']):
            answer_id = answers[idx]

            if answer_id >= self.answer_space_size:
                answer_id -= self.answer_space_size
                answer = word_tokenize(batch['ocr_tokens'][answer_id][idx])
            else:
                answer = self.answer_dict.idx2word(answer_id)

            predictions.append({
                'question_id': question_id.item(),
                'answer': answer
            })

        return predictions

    def load_item(self, idx):
        sample = super(TextVQADataset, self).load_item(idx)

        if sample['question_id'] is None:
            sample['question_id'] = -1
        return sample

    def verbose_dump(self, output, expected_output, info):
        return
