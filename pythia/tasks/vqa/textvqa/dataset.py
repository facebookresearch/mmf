from pythia.tasks.vqa.vizwiz import VizWizDataset


class TextVQADataset(VizWizDataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args,
                 **kwargs):
        super().__init__(dataset_type, imdb_file_index, config, *args,
                         **kwargs)

        self._name = "textvqa"

    def format_for_evalai(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx]

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[answer_id][idx]
            else:
                answer = self.answer_processor.idx2word(answer_id)

            predictions.append({
                'question_id': question_id.item(),
                'answer': answer
            })

        return predictions
