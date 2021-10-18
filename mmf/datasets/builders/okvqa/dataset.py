# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Type, Union

import torch
from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.builders.okvqa.database import OKVQAAnnotationDatabase
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.datasets.processors import GraphVQAAnswerProcessor


class OKVQADataset(MMFDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__("okvqa", config, dataset_type, index, *args, **kwargs)

    """def build_annotation_db(self) -> Type[OKVQAAnnotationDatabase]:
        annotation_path = self._get_path_based_on_index(
            self.config, "annotations", self._index
        )
        return OKVQAAnnotationDatabase(self.config, annotation_path)
    """

    def get_image_path(self, image_id: Union[str, int]) -> str:
        if self.dataset_type == "train":
            image_path = f"COCO_train2014_{str(image_id).zfill(12)}.jpg"
        else:
            image_path = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
        return image_path

    def init_processors(self):
        super().init_processors()
        if hasattr(self, "image_db"):
            self.image_db.transform = self.image_processor

    def __getitem__(self, idx: int) -> Type[Sample]:
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        if "question_tokens" in sample_info:
            text_processor_argument = {
                "tokens": sample_info["question_tokens"],
                "text": sample_info["question_str"],
            }
        else:
            text_processor_argument = {"text": sample_info["question"]}
        processed_question = self.text_processor(text_processor_argument)
        current_sample.update(processed_question)
        current_sample.id = torch.tensor(
            int(sample_info["question_id"]), dtype=torch.int
        )

        if self._use_features:
            features = self.features_db[idx]
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )
            current_sample.update(features)
        else:
            image_path = sample_info["image_name"] + ".jpg"
            current_sample.image = self.image_db.from_path(image_path)["images"][0]

        current_sample = self.add_answer_info(sample_info, current_sample)
        return current_sample

    def add_answer_info(self, sample_info, sample):
        if "answers" in sample_info:
            answers = sample_info["answers"]
            answer_processor_arg = {"answers": answers}
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)

            sample.targets = processed_soft_copy_answers["answers_scores"]

        return sample

    def idx_to_answer(self, idx):
        return self.answer_processor.convert_idx_to_answer(idx)

    def format_for_prediction(self, report):
        # Check for case of scores coming from graph
        reg_vocab_sz = self.answer_processor.get_true_vocab_size()
        if report.scores.size(1) > reg_vocab_sz:
            # Should actually have the graph_vqa_answer
            assert type(self.answer_processor.processor) is GraphVQAAnswerProcessor

            # Collapse into one set of confs (i.e. copy graph ones over if conf is greater)
            # Again, assumes graph ans is subset of all answers
            scores = torch.Tensor(report.scores.shape).copy_(report.scores)
            for batch_ind in range(report.scores.size(0)):
                for graph_ind, graph_ans in enumerate(
                    self.answer_processor.graph_vocab
                ):
                    # Get graph conf
                    graph_conf = scores[batch_ind, reg_vocab_sz + graph_ind].item()

                    # Get non-graph conf
                    reg_idx = self.answer_processor.answer_vocab.word2idx(graph_ans)
                    assert (
                        reg_idx != self.answer_processor.answer_vocab.UNK_INDEX
                        and reg_idx < reg_vocab_sz
                    )
                    reg_conf = scores[batch_ind, reg_idx].item()

                    # Set to max, zero out graph ind
                    scores[batch_ind, reg_idx] = max(graph_conf, reg_conf)
                    scores[batch_ind, reg_vocab_sz + graph_ind] = -float("Inf")
        else:
            scores = report.scores

        # Get top 5 answers and scores
        topkscores, topkinds = torch.topk(scores, 5, dim=1)

        answers = scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.id):
            # Dictionary to append for prediction
            pred_dict = {}
            pred_dict["question_id"] = question_id.item()

            # Get top-k answers
            assert (
                len(topkscores[idx]) == len(topkinds[idx]) and len(topkscores[idx]) == 5
            )
            topk_ans_scores = []
            for score, aid in zip(topkscores[idx], topkinds[idx]):
                score = score.item()
                kaid = aid.item()

                if kaid >= answer_space_size:
                    kaid -= answer_space_size
                    kanswer = report.context_tokens[idx][kaid]
                    if kanswer == self.context_processor.PAD_TOKEN:
                        kanswer = "unanswerable"
                else:
                    kanswer = self.answer_processor.idx2word(kaid)
                kanswer = kanswer.replace(" 's", "'s")
                topk_ans_scores.append((kanswer, score))
            pred_dict["topk"] = topk_ans_scores

            # Now get regular answer
            answer_id = answers[idx].item()
            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
                if answer == self.context_processor.PAD_TOKEN:
                    answer = "unanswerable"
            else:
                answer = self.answer_processor.idx2word(answer_id)

            answer = answer.replace(" 's", "'s")
            pred_dict["answer"] = answer
            predictions.append(pred_dict)

            # Dump the info
            info = {}
            info["scores"] = report.scores[idx].cpu()

        return predictions
