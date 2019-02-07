import torch

from collections import Counter

from pythia.tasks.datasets.vizwiz.dataset import VizWizDataset


class TextVQADataset(VizWizDataset):
    def __init__(self, imdb_file, image_feat_directories, verbose=False,
                 **data_params):
        super(TextVQADataset, self).__init__(imdb_file, image_feat_directories,
                                             verbose, **data_params)

        self.name = "textvqa"
        self.ocr_count_greater_than_2 = 0
        self.ocr_predicted_when_count_greater_than_2 = 0
        self.ocr_correct_predicted_when_count_greater_than_2 = 0

    def format_for_evalai(self, batch, answers):
        raise RuntimeError("TextVQA doesn't support EvalAI predict mode")

    def load_item(self, idx):
        sample = super(TextVQADataset, self).load_item(idx)

        if sample['question_id'] is None:
            sample['question_id'] = -1
        return sample

    def verbose_dump(self, output, expected_output, info):
        # TODO: Reformat and log to writer also

        # print("OCR Length")
        # print(info['original_batch']['context_dim'])
        print("Valid Answers")
        print(info['original_batch']['valid_answers'])
        # print("OCR Attentions")
        # print(info['context_attentions'])

        answer_counter = Counter()
        # print("Expected answers")
        # print(info['original_batch']['answers'])
        actual = torch.max(output, 1)[1].data  # argmax

        for idx, item in enumerate(actual):
            answer_counter.clear()
            for i in info['original_batch']['valid_answers']:
                answer_counter[i[idx]] += 1

            ocr_count = 0

            for i in info['original_batch']['ocr_tokens']:
                if i[idx] == 'unanswerable':
                    break
                ocr_count += 1

            if ocr_count >= 2:
                self.ocr_count_greater_than_2 += 1
            item = item.item()
            self.num_times_predicted += 1
            if item >= self.answer_space_size:
                self.num_times_predicted_from_ocr += 1
                if ocr_count >= 2:
                    self.ocr_predicted_when_count_greater_than_2 += 1
                print("Predicted from OCR tokens")
                tokens = info['original_batch']['ocr_tokens']
                prediction = tokens[item - self.answer_space_size][idx]
                if answer_counter[prediction] >= 3:
                    self.correct_predicted_from_ocr += 1
                    if ocr_count >= 2:
                        self.ocr_correct_predicted_when_count_greater_than_2 \
                            += 1
                else:
                    partial_count = 0
                    for key, count in answer_counter.items():
                        if key.split(' ').count(prediction) > 0:
                            partial_count += count

                    if partial_count >= 3:
                        self.partial_predicted_from_ocr += 1
                print(prediction)
            else:
                self.num_times_predicted_from_as += 1
                print("Predicted from answer space")
                prediction = self.answer_dict.idx2word(item)

                if answer_counter[prediction] >= 3:
                    self.correct_predicted_from_as += 1
                print(prediction)
        print("OCR Count greater than 2", self.ocr_count_greater_than_2,
              "/", self.num_times_predicted, ",",
              self.ocr_count_greater_than_2 / self.num_times_predicted)
        print("OCR Count greater than 2 Choose to Copy",
              self.ocr_predicted_when_count_greater_than_2,
              "/", self.ocr_count_greater_than_2, ",",
              self.ocr_predicted_when_count_greater_than_2 /
              self.ocr_count_greater_than_2)
        print("OCR Count greater than 2 Correct Copy",
              self.ocr_correct_predicted_when_count_greater_than_2,
              "/", self.ocr_predicted_when_count_greater_than_2, ",",
              self.ocr_correct_predicted_when_count_greater_than_2 /
              self.ocr_predicted_when_count_greater_than_2)

        print("% predicted from OCR: ",
              self.num_times_predicted_from_ocr / self.num_times_predicted)
        print("% predicted from Answer space: ",
              self.num_times_predicted_from_as / self.num_times_predicted)
        if self.num_times_predicted_from_ocr > 0:
            print("% correct predicted from OCR: ",
                  self.correct_predicted_from_ocr
                  / self.num_times_predicted_from_ocr)
            print("% partial predicted from OCR: ",
                  self.partial_predicted_from_ocr
                  / self.num_times_predicted_from_ocr)
        if self.num_times_predicted_from_as > 0:
            print("% correct predicted from Answer space: ",
                  self.correct_predicted_from_as
                  / self.num_times_predicted_from_as)
