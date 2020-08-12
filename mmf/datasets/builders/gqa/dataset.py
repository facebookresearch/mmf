# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset


class GQADataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__("gqa", config, dataset_type, imdb_file_index, *args, **kwargs)

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        text_processor_argument = {"text": sample_info["question_str"]}
        processed_question = self.text_processor(text_processor_argument)
        current_sample.text = processed_question["text"]
        if "input_ids" in processed_question:
            current_sample.update(processed_question)

        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["image_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["image_id"]

        if self._use_features is True:
            features = self.features_db[idx]
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )
            current_sample.update(features)

        # Depending on whether we are using soft copy this can add
        # dynamic answer space
        current_sample = self.add_answer_info(sample_info, current_sample)
        return current_sample

    def add_answer_info(self, sample_info, sample):
        if "answers" in sample_info:
            answers = sample_info["answers"]
            answer_processor_arg = {"answers": answers}
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)
            sample.targets = processed_soft_copy_answers["answers_scores"]

        return sample

    def format_for_prediction(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
                if answer == self.context_processor.PAD_TOKEN:
                    answer = "unanswerable"
            else:
                answer = self.answer_processor.idx2word(answer_id)

            predictions.append({"questionId": question_id.item(), "prediction": answer})

        return predictions
