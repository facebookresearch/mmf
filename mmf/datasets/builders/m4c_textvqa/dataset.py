# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch

from mmf.common.sample import Sample
from mmf.datasets.builders.textvqa.dataset import TextVQADataset
from mmf.utils.objects_to_byte_tensor import dec_bytes2obj, enc_obj2bytes
from mmf.utils.text import word_tokenize


class M4CTextVQADataset(TextVQADataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "m4c_textvqa"

    def preprocess_sample_info(self, sample_info):
        return sample_info  # Do nothing

    def postprocess_evalai_entry(self, entry):
        return entry  # Do nothing

    def format_for_evalai(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        pred_answers = report.scores.argmax(dim=-1).view(batch_size, -1)
        answer_space_size = answer_processor.get_true_vocab_size()

        image_id_enc = report.image_id_enc.cpu().numpy()
        context_tokens_enc = report.context_tokens_enc.cpu().numpy()
        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            image_id = dec_bytes2obj(image_id_enc[idx])
            context_tokens = dec_bytes2obj(context_tokens_enc[idx])
            answer_words = []
            pred_source = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(word_tokenize(context_tokens[answer_id]))
                    pred_source.append("OCR")
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )
                    pred_source.append("VOCAB")
            # join all the answer tokens with space
            # (this should be correct for almost all cases)
            pred_answer = " ".join(answer_words).replace(" 's", "'s")
            entry = {
                "question_id": question_id.item(),
                "image_id": image_id,
                "answer": pred_answer,
                "pred_source": pred_source,
            }
            entry = self.postprocess_evalai_entry(entry)

            predictions.append(entry)

        return predictions

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        sample_info = self.preprocess_sample_info(sample_info)
        current_sample = Sample()

        # breaking change from VQA2Dataset: load question_id
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = str(sample_info["image_id"])
        else:
            current_sample.image_id = sample_info["image_id"]

        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)

        current_sample = self.add_sample_details(sample_info, current_sample)
        current_sample = self.add_answer_info(sample_info, current_sample)

        # only the 'max_features' key is needed
        # pop other keys to minimize data loading overhead
        for k in list(current_sample.image_info_0):
            if k != "max_features":
                current_sample.image_info_0.pop(k)
        for k in list(current_sample.image_info_1):
            if k != "max_features":
                current_sample.image_info_1.pop(k)

        return current_sample

    def add_sample_details(self, sample_info, sample):
        sample.image_id_enc = enc_obj2bytes(sample_info["image_id"])

        # 1. Load text (question words)
        # breaking change from VQA2Dataset:
        # load the entire question string, not tokenized questions, since we
        # switch to BERT tokenizer in M4C and do online tokenization
        question_str = (
            sample_info["question"]
            if "question" in sample_info
            else sample_info["question_str"]
        )
        processed_question = self.text_processor({"question": question_str})
        sample.text = processed_question["token_inds"]
        sample.text_len = processed_question["token_num"]

        # 2. Load object
        # object bounding box information
        sample.obj_bbox_coordinates = self.copy_processor(
            {"blob": sample_info["obj_normalized_boxes"]}
        )["blob"]

        # 3. Load OCR
        if not self.use_ocr:
            # remove all OCRs from the sample
            # (i.e. make an empty OCR list)
            sample_info["ocr_tokens"] = []
            sample_info["ocr_info"] = []
            if "ocr_normalized_boxes" in sample_info:
                sample_info["ocr_normalized_boxes"] = np.zeros((0, 4), np.float32)
            # clear OCR visual features
            sample.image_feature_1 = torch.zeros_like(sample.image_feature_1)

        # Preprocess OCR tokens
        ocr_tokens = [
            self.ocr_token_processor({"text": token})["text"]
            for token in sample_info["ocr_tokens"]
        ]
        # Get FastText embeddings for OCR tokens
        context = self.context_processor({"tokens": ocr_tokens})
        sample.context = context["text"]
        sample.context_tokens = context["tokens"]
        sample.context_tokens_enc = enc_obj2bytes(context["tokens"])
        sample.context_feature_0 = context["text"]
        sample.context_info_0 = Sample()
        sample.context_info_0.max_features = context["length"]
        # Get PHOC embeddings for OCR tokens
        context_phoc = self.phoc_processor({"tokens": ocr_tokens})
        sample.context_feature_1 = context_phoc["text"]
        sample.context_info_1 = Sample()
        sample.context_info_1.max_features = context_phoc["length"]
        # OCR order vectors
        # TODO remove order_vectors -- it is no longer needed in M4C
        order_vectors = np.eye(len(sample.context_tokens), dtype=np.float32)
        order_vectors = torch.from_numpy(order_vectors)
        order_vectors[context["length"] :] = 0
        sample.order_vectors = order_vectors
        # OCR bounding box information
        if "ocr_normalized_boxes" in sample_info:
            # New imdb format: OCR bounding boxes are already pre-computed
            max_len = self.config.processors.answer_processor.params.max_length
            sample.ocr_bbox_coordinates = self.copy_processor(
                {"blob": sample_info["ocr_normalized_boxes"]}
            )["blob"][:max_len]
        else:
            # Old imdb format: OCR bounding boxes are computed on-the-fly
            # from ocr_info
            sample.ocr_bbox_coordinates = self.bbox_processor(
                {"info": sample_info["ocr_info"]}
            )["bbox"].coordinates

        return sample

    def add_answer_info(self, sample_info, sample):
        sample_has_answer = "answers" in sample_info
        if sample_has_answer:
            # Load real answers from sample_info
            answers = sample_info["answers"]
            sample.gt_answers_enc = enc_obj2bytes(answers)
            answer_processor_arg = {
                "answers": answers,
                "context_tokens": sample.context_tokens,
            }
            processed_answers = self.answer_processor(answer_processor_arg)

            assert not self.config.fast_read, (
                "In M4CTextVQADataset, online OCR sampling is incompatible "
                "with fast_read, so fast_read is currently not supported."
            )
            sample.targets = processed_answers["answers_scores"]
            sample.sampled_idx_seq = processed_answers["sampled_idx_seq"]
            sample.train_prev_inds = processed_answers["train_prev_inds"]
            sample.train_loss_mask = processed_answers["train_loss_mask"]
        else:
            # Load dummy answers as placeholders
            answer_params = self.config.processors.answer_processor.params
            sample.sampled_idx_seq = None
            sample.train_prev_inds = torch.zeros(
                answer_params.max_copy_steps, dtype=torch.long
            )

        return sample
