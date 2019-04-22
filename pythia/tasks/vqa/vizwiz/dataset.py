# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from pythia.tasks.vqa.vqa2 import VQA2Dataset
from pythia.common.sample import Sample


class VizWizDataset(VQA2Dataset):
    def __init__(self, dataset_type, imdb_file_index, config,
                 *args, **kwargs):
        super().__init__(dataset_type, imdb_file_index, config,
                         *args, **kwargs)

        # Update name as default would be 'vqa2' due to inheritance
        self._name = 'vizwiz'

        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info

    def load_item(self, idx):
        sample = super().load_item(idx)

        sample_info = self.imdb[idx]

        if "image_name" in sample_info:
            sample.image_id = sample_info["image_name"]

        if self.use_ocr:
            # Preprocess OCR tokens
            ocr_tokens = [self.ocr_token_processor({"text": token})["text"]
                          for token in sample_info["ocr_tokens"]]
            # Get embeddings for tokens
            context = self.context_processor({
                "tokens": ocr_tokens
            })
            sample.context = context["text"]
            sample.context_tokens = context["tokens"]
            sample.context_feature_0 = context["text"]
            sample.context_info_0 = Sample()
            sample.context_info_0.max_features = context["length"]

            order_vectors = torch.eye(len(sample.context_tokens))
            order_vectors[context["length"]:] = 0
            sample.order_vectors = order_vectors

        if self.use_ocr_info and "ocr_info" in sample_info:
            sample.ocr_bbox = self.bbox_processor({
                "info": sample_info["ocr_info"]
            })["bbox"]
        return sample

    def add_answer_info(self, sample_info, sample):
        if "answers" in sample_info:
            answers = sample_info["answers"]
            processed_soft_copy_answers = self.answer_processor({
                "answers": answers,
                "tokens": sample_info["ocr_tokens"]
            })

            sample.answers = processed_soft_copy_answers["answers"]
            sample.targets = processed_soft_copy_answers["answers_scores"]

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
            predictions.append({
                'image': "_".join(["VizWiz"] + image_id.split("_")[2:])
                         + ".jpg",
                'answer': answer
            })

        return predictions
