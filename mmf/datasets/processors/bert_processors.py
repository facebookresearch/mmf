# Copyright (c) Facebook, Inc. and its affiliates.

import random

import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.datasets.processors.processors import BaseProcessor
from transformers.tokenization_auto import AutoTokenizer


@registry.register_processor("masked_token")
class MaskedTokenProcessor(BaseProcessor):
    _CLS_TOKEN = "[CLS]"
    _SEP_TOKEN = "[SEP]"

    def __init__(self, config, *args, **kwargs):
        tokenizer_config = config.tokenizer_config
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config.type, **tokenizer_config.params
        )

        self._max_seq_length = config.max_seq_length
        self._probability = getattr(config, "mask_probability", 0.15)

    def get_vocab_size(self):
        return len(self._tokenizer)

    def tokenize(self, tokens):
        return self._tokenizer.tokenize(tokens)

    def _random_word(self, tokens, probability=0.15):
        labels = []
        for idx, token in enumerate(tokens):
            prob = random.random()

            if prob < probability:
                prob /= probability

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[idx] = "[MASK]"
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[idx] = self._tokenizer.convert_ids_to_tokens(
                        torch.randint(len(self._tokenizer), (1,), dtype=torch.long)
                    )[0]

                # rest 10% keep the original token as it is

                labels.append(self._tokenizer.convert_tokens_to_ids(token))
            else:
                labels.append(-1)

        return tokens, labels

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        if tokens_b is None:
            tokens_b = []
        else:
            # _convert_to_indices does [CLS] tokens_a [SEP] tokens_b [SEP]
            max_length -= 1
            assert max_length >= 0, (
                "Max length should be minimum 2 in case of single sentence"
                + " and 3 in case of two sentences."
            )

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_to_indices(self, tokens_a, tokens_b=None, probability=0.15):
        tokens_a, label_a = self._random_word(tokens_a, probability=probability)
        tokens = [self._CLS_TOKEN] + tokens_a + [self._SEP_TOKEN]
        segment_ids = [0] + [0] * len(tokens_a) + [0]

        if tokens_b:
            tokens_b, label_b = self._random_word(tokens_b, probability=probability)
            lm_label_ids = [-1] + label_a + [-1] + label_b + [-1]
            assert len(tokens_b) > 0
            tokens += tokens_b + [self._SEP_TOKEN]
            segment_ids += [1] * len(tokens_b) + [1]
        else:
            lm_label_ids = [-1] + label_a + [-1]

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self._max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == self._max_seq_length
        assert len(input_mask) == self._max_seq_length
        assert len(segment_ids) == self._max_seq_length
        assert len(lm_label_ids) == self._max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        lm_label_ids = torch.tensor(lm_label_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "lm_label_ids": lm_label_ids,
            "tokens": tokens,
        }

    def __call__(self, item):
        text_a = item["text_a"]
        text_b = item.get("text_b", None)

        tokens_a = self.tokenize(text_a)
        tokens_b = None

        if text_b:
            tokens_b = self.tokenize(text_b)

        self._truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length - 2)
        output = self._convert_to_indices(
            tokens_a, tokens_b, probability=self._probability
        )
        output["is_correct"] = torch.tensor(item["is_correct"], dtype=torch.long)

        return output


@registry.register_processor("bert_tokenizer")
class BertTokenizer(MaskedTokenProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._probability = 0

    def __call__(self, item):

        if "text" in item:
            text_a = item["text"]
        else:
            text_a = " ".join(item["tokens"])

        if isinstance(text_a, list):
            text_a = " ".join(text_a)

        tokens_a = self.tokenize(text_a)

        # 'text_b' can be defined in the dataset preparation
        tokens_b = None
        if "text_b" in item:
            text_b = item["text_b"]
            if text_b:
                tokens_b = self.tokenize(text_b)

        self._truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length - 2)
        output = self._convert_to_indices(
            tokens_a, tokens_b, probability=self._probability
        )
        output["text"] = output["tokens"]
        return output


@registry.register_processor("multi_sentence_bert_tokenizer")
class MultiSentenceBertTokenizer(BertTokenizer):
    """Extension of BertTokenizer which supports multiple sentences.
    Separate from normal usecase, each sentence will be passed through
    bert tokenizer separately and indices will be reshaped as single
    tensor. Segment ids will also be increasing in number.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.fusion_strategy = config.get("fusion", "concat")
        self.tokenizer = super().__call__

    def __call__(self, item):
        texts = item["text"]
        if not isinstance(texts, list):
            texts = [texts]

        processed = []
        for idx, text in enumerate(texts):
            sample = Sample()
            processed_text = self.tokenizer({"text": text})
            sample.update(processed_text)
            sample.segment_ids.fill_(idx)
            processed.append(sample)
        # Use SampleList to convert list of tensors to stacked tensors
        processed = SampleList(processed)
        if self.fusion_strategy == "concat":
            processed.input_ids = processed.input_ids.view(-1)
            processed.input_mask = processed.input_mask.view(-1)
            processed.segment_ids = processed.segment_ids.view(-1)
            processed.lm_label_ids = processed.lm_label_ids.view(-1)
        return processed.to_dict()
