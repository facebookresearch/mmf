import random

from pythia.common.sample import Sample
from pythia.datasets.vqa.vqa2.dataset import VQA2Dataset


class MaskedVQA2Dataset(VQA2Dataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(dataset_type, imdb_file_index, config, *args, **kwargs)
        self._name = "masked_vqa2"
        self._add_answer = config.get("add_answer", True)

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        current_sample = Sample()

        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)

        current_sample = self._add_masked_question(sample_info, current_sample)

        return current_sample

    def _add_masked_question(self, sample_info, current_sample):
        question = sample_info["question_str"]
        random_answer = random.choice(sample_info["all_answers"])

        processed = self.masked_token_processor({
            "text_a": question,
            "text_b": random_answer,
            "is_correct": -1
        })

        processed.pop("tokens")
        current_sample.update(processed)

        return current_sample
