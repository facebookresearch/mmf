import random

from mmf.datasets.builders.vqa2.dataset import VQA2Dataset


class MaskedQVQA2Dataset(VQA2Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = "masked_q_vqa2"

    def add_answer_info(self, sample_info, current_sample):
        length = min(len(current_sample.text), current_sample.text_len)
        index = random.randint(0, length - 1)

        word = self.text_processor.vocab.get_itos()[current_sample.text[index].item()]
        current_sample.text[index] = self.text_processor.vocab.get_stoi()["<mask>"]
        answer_processor_arg = {"answers": [word]}

        processed_soft_copy_answers = self.answer_processor(answer_processor_arg)

        current_sample.answers = processed_soft_copy_answers["answers"]
        current_sample.targets = processed_soft_copy_answers["answers_scores"]

        if self.answer_processor.word2idx(word) == self.answer_processor.word2idx(
            "<unk>"
        ):
            current_sample.targets.zero_()
        return current_sample
