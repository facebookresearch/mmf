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

        output = self._prepare_processed_output(
            input_ids, input_mask, segment_ids, lm_label_ids
        )
        return {**output, "tokens": tokens}

    def _prepare_processed_output(
        self, input_ids, input_mask, segment_ids, lm_label_ids
    ):
        assert (
            len(input_ids) == len(input_mask) == len(segment_ids) == len(lm_label_ids)
        )
        input_ids += (self._max_seq_length - len(input_ids)) * [0]
        input_mask += (self._max_seq_length - len(input_mask)) * [0]
        segment_ids += (self._max_seq_length - len(segment_ids)) * [0]
        lm_label_ids += (self._max_seq_length - len(lm_label_ids)) * [-1]

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


@registry.register_processor("masked_multi_sentence_bert_tokenizer")
class MaskedMultiSentenceBertTokenizer(BertTokenizer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.fusion_strategy = config.get("fusion", "concat")
        self._probability = config.get("mask_probability", 0.15)
        self._type = config.get("type", None)
        self._with_sentence_seperator = config.get("with_sentence_separator", False)

    def __call__(self, item):
        sentences = item["sentences"]
        sentences, sentence_tokens = self._tokenize_sentences(sentences)
        assert len(sentences) == len(sentence_tokens)

        if "two_sentences" in self._type:
            return self._two_sentences(sentence_tokens)
        elif "rand_sentences" in self._type:
            return self._rand_sentences(sentence_tokens)
        else:  # self._type == "full_sentences":
            return self._full_sentences(sentence_tokens)

    def _tokenize_sentences(self, texts):
        sentences, sentence_tokens = [], []
        for sentence in texts:
            sentence = sentence.strip()
            tokens = self._tokenizer.tokenize(sentence)
            sentences.append(sentence)
            sentence_tokens.append(tokens)
        return sentences, sentence_tokens

    def _two_sentences(self, sentence_tokens):
        """Concatnate only two sentences starting with either 0 or random index.
        With optional separator. Truncate or else pad to max_seq_length.
        """
        has_two_or_more_sentences = len(sentence_tokens) > 1
        if "rand" in self._type:
            start_idx = (
                random.randint(0, len(sentence_tokens) - 2)
                if has_two_or_more_sentences
                else 0
            )
        else:
            start_idx = 0
        end_idx = start_idx + 2 if has_two_or_more_sentences else len(sentence_tokens)
        return self._full_sentences(sentence_tokens[start_idx:end_idx])

    def _rand_sentences(self, sentence_tokens):
        """Concatnate all consecutive sentences with optional separator.
        Start with a random sentence index aiming to fill max_seq_length.
        Pad till max_seq_length.
        """
        sentence_separater = self._sentence_separator("sentence")

        # max_start_idx is the upper bound of the random index generator with the goal
        # to freducing the need for padding as much as possible. Note: padding may still
        # be necessary if total length is smaller than max_seq_length
        max_start_idx = 0
        length_to_fullfill = self._max_seq_length - 2
        # rest_token_length is the token length from idx;
        # It starts as the length + seperators
        rest_token_length = (
            sum(len(x) for x in sentence_tokens) + len(sentence_tokens) - 1
        )
        for idx, sentence_token in enumerate(sentence_tokens):
            if (
                rest_token_length - len(sentence_token) - len(sentence_separater)
                < length_to_fullfill
            ):
                max_start_idx = idx
                break
            rest_token_length -= len(sentence_token) + 1

        random_idx = random.randint(0, max_start_idx - 1 if max_start_idx > 0 else 0)
        return self._full_sentences(sentence_tokens[random_idx:])

    def _full_sentences(self, sentence_tokens):
        """Concatnate all sentences starting with sentence index 0.
        With optional separator. Truncate or else pad to max_seq_length.
        """
        tokens = []
        lm_label_ids = []
        for idx, sentence_token in enumerate(sentence_tokens):
            token_a, label_a = self._random_word(
                sentence_token, probability=self._probability
            )
            if idx == len(sentence_tokens) - 1:
                tokens += token_a
                lm_label_ids += label_a
            else:
                tokens += token_a + self._sentence_separator("sentence")
                lm_label_ids += label_a + self._sentence_separator("id")

        tokens = tokens[: self._max_seq_length - 2]
        tokens = [self._CLS_TOKEN] + tokens + [self._SEP_TOKEN]
        lm_label_ids = lm_label_ids[: self._max_seq_length - 2]
        lm_label_ids = [-1] + lm_label_ids + [-1]
        segment_ids = [0] * len(tokens)
        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        output = self._prepare_processed_output(
            input_ids, input_mask, segment_ids, lm_label_ids
        )
        return {
            **output,
            "tokens": tokens,
            "is_correct": torch.tensor(True, dtype=torch.long),
        }

    def _sentence_separator(self, key):
        if self._with_sentence_seperator:
            return {"sentence": [self._SEP_TOKEN], "id": [-1]}[key]
        else:
            return []
