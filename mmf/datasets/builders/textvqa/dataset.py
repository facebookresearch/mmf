# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.datasets.builders.vizwiz import VizWizDataset
from mmf.utils.text import word_tokenize


class TextVQADataset(VizWizDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "textvqa"

    def format_for_evalai(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx].item()
            print(
                answer_id,
                idx,
                len(answers),
                len(report.question_id),
                len(report.context_tokens),
            )
            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = word_tokenize(report.context_tokens[idx][answer_id])
            else:
                answer = self.answer_processor.idx2word(answer_id)

            predictions.append({"question_id": question_id.item(), "answer": answer})
        return predictions
