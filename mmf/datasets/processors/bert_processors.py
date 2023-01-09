# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.datasets.processors.processors import BaseProcessor

try:
    from transformers3.tokenization_auto import AutoTokenizer
except ImportError:
    from transformers.tokenization_auto import AutoTokenizer


@registry.register_processor("masked_token")
class MaskedTokenProcessor(BaseProcessor):
    _CLS_TOKEN = "[CLS]"
    _SEP_TOKEN = "[SEP]"
    _MASK_TOKEN = "[MASK]"
    _PAD_TOKEN_ID = 0

    def __init__(self, config, *args, **kwargs):
        tokenizer_config = config.tokenizer_config
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config.type, **tokenizer_config.params
        )

        self._max_seq_length = config.max_seq_length
        self._probability = config.get("mask_probability", 0.15)

    def get_vocab_size(self) -> int:
        return len(self._tokenizer)

    def tokenize(self, tokens: Union[str, List[str]]) -> List[str]:
        return self._tokenizer.tokenize(tokens)

    def _convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def _convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        return self._tokenizer.convert_ids_to_tokens(ids)

    def _random_word(
        self, tokens: List[str], probability: float = 0.15
    ) -> Tuple[List[str], List[int]]:
        labels = []
        for idx, token in enumerate(tokens):
            prob = random.random()

            if prob < probability:
                prob /= probability

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[idx] = self._MASK_TOKEN
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[idx] = self._convert_ids_to_tokens(
                        torch.randint(self.get_vocab_size(), (1,), dtype=torch.long)
                    )[0]

                # rest 10% keep the original token as it is

                labels.append(self._convert_tokens_to_ids(token))
            else:
                labels.append(-1)

        return tokens, labels

    def _truncate_seq_pair(
        self, tokens_a: List[str], tokens_b: List[str], max_length: int
    ):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        if tokens_b is None:
            tokens_b = []
            max_length -= 2
        else:
            # _convert_to_indices does [CLS] tokens_a [SEP] tokens_b [SEP]
            max_length -= 3
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

    def _convert_to_indices(
        self,
        tokens_a: List[str],
        tokens_b: Optional[List[str]] = None,
        probability: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        """
        BERT encodes
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        """
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

        input_ids = self._convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self._max_seq_length:
            input_ids.append(self._PAD_TOKEN_ID)
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

    def __call__(self, item: Dict[str, Any]):
        text_a = item["text_a"]
        text_b = item.get("text_b", None)

        tokens_a = self.tokenize(text_a)
        tokens_b = None

        if text_b:
            tokens_b = self.tokenize(text_b)

        self._truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length)
        output = self._convert_to_indices(
            tokens_a, tokens_b, probability=self._probability
        )
        output["is_correct"] = torch.tensor(
            item.get("is_correct", True), dtype=torch.long
        )

        return output


@registry.register_processor("bert_tokenizer")
class BertTokenizer(MaskedTokenProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._probability = config.get("mask_probability", 0)

    def __call__(self, item: Dict[str, Any]):
        if "text" in item:
            text_a = item["text"]
        elif "text_a" in item:
            text_a = item["text_a"]
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

        self._truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length)
        output = self._convert_to_indices(
            tokens_a, tokens_b, probability=self._probability
        )
        output["text"] = output["tokens"]
        return output


@registry.register_processor("multi_sentence_bert_tokenizer")
class MultiSentenceBertTokenizer(BaseProcessor):
    """Extension of BertTokenizer which supports multiple sentences.
    Separate from normal usecase, each sentence will be passed through
    bert tokenizer separately and indices will be reshaped as single
    tensor. Segment ids will also be increasing in number.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.fusion_strategy = config.get("fusion", "concat")
        self._probability = config.get("mask_probability", 0)
        self.tokenizer = BertTokenizer(config)

    def __call__(self, item: Dict[str, Any]):
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


@registry.register_processor("masked_roberta_tokenizer")
class MaskedRobertaTokenizer(MaskedTokenProcessor):
    def __init__(self, config, *args, **kwargs):
        # https://huggingface.co/transformers/model_doc/xlmroberta.html
        # roberta is with different tokenization of above default (bert)
        tokenizer_config = config.tokenizer_config
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config.type, **tokenizer_config.params
        )

        self._CLS_TOKEN = self._tokenizer.bos_token  # <s>
        self._SEP_TOKEN = self._tokenizer.sep_token  # </s>
        self._MASK_TOKEN = self._tokenizer.mask_token  # <mask>
        self._PAD_TOKEN_ID = self._tokenizer.pad_token_id  # 1

        self._max_seq_length = config.max_seq_length
        self._probability = getattr(config, "mask_probability", 0.15)

    def _truncate_seq_pair(
        self, tokens_a: List[str], tokens_b: List[str], max_length: int
    ):
        """Truncates a sequence pair in place to the maximum length."""
        if tokens_b is None:
            tokens_b = []
            max_length -= 2
        else:
            # _convert_to_indices does <s> tokens_a </s> </s> tokens_b </s>
            max_length -= 4
        assert max_length >= 0, (
            "Max length should be minimum 2 in case of single sentence"
            + " and 4 in case of two sentences."
        )

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_to_indices(
        self,
        tokens_a: List[str],
        tokens_b: Optional[List[str]] = None,
        probability: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        """
        Roberta encodes
        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s> </s> B </s>``
        """
        tokens_a, label_a = self._random_word(tokens_a, probability=probability)
        tokens = [self._CLS_TOKEN] + tokens_a + [self._SEP_TOKEN]
        segment_ids = [0] + [0] * len(tokens_a) + [0]
        lm_label_ids = [-1] + label_a + [-1]

        if tokens_b:
            tokens_b, label_b = self._random_word(tokens_b, probability=probability)
            assert len(tokens_b) > 0
            # ``<s> A </s> </s> B </s>``
            tokens += [self._SEP_TOKEN] + tokens_b + [self._SEP_TOKEN]
            # RoBERTA and XLM-R don't use segment_ids, segment_ids are all 0's
            segment_ids += [0] + [0] * len(tokens_b) + [0]
            lm_label_ids += [-1] + label_b + [-1]

        input_ids = self._convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self._max_seq_length:
            input_ids.append(self._PAD_TOKEN_ID)
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


@registry.register_processor("roberta_tokenizer")
class RobertaTokenizer(BertTokenizer, MaskedRobertaTokenizer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._probability = config.get("mask_probability", 0)


@registry.register_processor("multi_sentence_roberta_tokenizer")
class MultiSentenceRobertaTokenizer(MultiSentenceBertTokenizer):
    """Extension of SPMTokenizer which supports multiple sentences.
    Similar to MultiSentenceBertTokenizer.
    """

    def __init__(self, config, *args, **kwargs):
        self.fusion_strategy = config.get("fusion", "concat")
        self.tokenizer = RobertaTokenizer(config, *args, **kwargs)
        self._probability = config.get("mask_probability", 0)


def get_pair_text_tokens(
    item: Dict[str, Any], masked_token_processor: MaskedTokenProcessor
) -> Dict[str, torch.Tensor]:
    """Given an item Dict with either 1 or 2 text sentences,
    tokenize and concat them returning a Dict that contains at least
    "input_ids", "input_mask", and "segment_ids'.

    Args:
        item (Dict[str, Any]):
            A Dict containing keys
            "text" or "tokens", and optionally "text_b"
        masked_token_processor (MaskedTokenProcessor):
            A processor used to tokenize the texts.

    Returns:
        [Dict[str, torch.Tensor]]:
            A Dict containing tokenized texts and
            related tensors.
    """
    if "text" in item:
        text_a = item["text"]
    elif "text_a" in item:
        text_a = item["text_a"]
    else:
        text_a = " ".join(item["tokens"])

    if isinstance(text_a, list):
        text_a = " ".join(text_a)

    tokens_a = masked_token_processor.tokenize(text_a)

    # 'text_b' can be defined in the dataset preparation
    tokens_b = None
    if "text_b" in item:
        text_b = item["text_b"]
        if text_b:
            tokens_b = masked_token_processor.tokenize(text_b)

    masked_token_processor._truncate_seq_pair(
        tokens_a, tokens_b, masked_token_processor._max_seq_length
    )
    output = masked_token_processor._convert_to_indices(
        tokens_a, tokens_b, probability=masked_token_processor._probability
    )
    return output


@registry.register_processor("vilt_text_tokenizer")
class VILTTextTokenizer(MaskedTokenProcessor):
    def __init__(self, config, *args, **kwargs):
        try:
            from transformers3 import BertTokenizer
        except ImportError:
            from transformers import BertTokenizer

        if isinstance(config, str):
            config = {"from_pretrained": config}

        from_pretrained_name = config.get("from_pretrained", "bert-base-uncased")
        kwargs_dict = dict(kwargs, do_lower_case="uncased" in from_pretrained_name)
        self._tokenizer = BertTokenizer.from_pretrained(
            from_pretrained_name, **kwargs_dict
        )
        self._max_seq_length = config.get("max_seq_length", 25)
        self._probability = config.get("mask_probability", 0)

    def __call__(self, item):
        output = get_pair_text_tokens(item, self)
        output["text"] = output["tokens"]
        return output


@registry.register_processor("uniter_text_tokenizer")
class UNITERTextTokenizer(MaskedTokenProcessor):
    def __init__(self, config, *args, **kwargs):
        try:
            from transformers3 import BertTokenizer
        except ImportError:
            from transformers import BertTokenizer

        if isinstance(config, str):
            config = {"from_pretrained": config}

        from_pretrained_name = config.get("from_pretrained", "bert-base-uncased")
        kwargs_dict = dict(kwargs, do_lower_case="uncased" in from_pretrained_name)
        self._tokenizer = BertTokenizer.from_pretrained(
            from_pretrained_name, **kwargs_dict
        )
        self._max_seq_length = config.get("max_seq_length", 25)
        self._probability = config.get("mask_probability", 0)

    def __call__(self, item: Dict[str, Any]):
        output = get_pair_text_tokens(item, self)
        output["text"] = output["tokens_masked"]
        output["tokens"] = output["tokens_masked"]
        if "is_correct" in item:
            output["is_correct"] = torch.tensor(
                item.get("is_correct", True), dtype=torch.long
            )
        return output

    def _token_transform(
        self, tokens: List[str], tokens_b: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, int, int, List[str]]:
        tokens = [self._CLS_TOKEN] + tokens + [self._SEP_TOKEN]
        if tokens_b:
            tokens += tokens_b + [self._SEP_TOKEN]

        input_ids = self._convert_tokens_to_ids(tokens)
        token_len = len(input_ids)
        token_pad = self._max_seq_length - token_len
        # Zero-pad up to the sequence length.
        input_ids += [self._PAD_TOKEN_ID] * token_pad
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        return input_ids_tensor, token_len, token_pad, tokens

    def _convert_to_indices(
        self,
        tokens_a: List[str],
        tokens_b: Optional[List[str]] = None,
        probability: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        """
        BERT encodes
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        """
        input_ids_original, _, _, _ = self._token_transform(tokens_a, tokens_b)

        tokens_a, label_a = self._random_word(tokens_a, probability=probability)
        segment_ids = [0] * (len(tokens_a) + 2)

        if tokens_b:
            tokens_b, label_b = self._random_word(tokens_b, probability=probability)
            lm_label_ids = [-1] + label_a + [-1] + label_b + [-1]
            assert len(tokens_b) > 0
            segment_ids += [1] * (len(tokens_b) + 1)
        else:
            lm_label_ids = [-1] + label_a + [-1]

        input_ids_masked, token_len, token_pad, tokens_masked = self._token_transform(
            tokens_a, tokens_b
        )

        input_mask = [1] * token_len + [0] * token_pad
        segment_ids += [0] * token_pad
        lm_label_ids += [-1] * token_pad

        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        lm_label_ids = torch.tensor(lm_label_ids, dtype=torch.long)
        return {
            "input_ids_masked": input_ids_masked,  # specifically for MLM heads
            "input_ids": input_ids_original,  # unmasked tokens for CLIP heads
            # input_mask is non-padding (1) vs padding (0) mask (not MLM token masking)
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "lm_label_ids": lm_label_ids,
            "tokens_masked": tokens_masked,
        }


@registry.register_processor("vinvl_text_tokenizer")
class VinVLTextTokenizer(MaskedTokenProcessor):
    def __init__(self, config, *args, **kwargs):
        try:
            from transformers3 import BertTokenizer
        except ImportError:
            from transformers import BertTokenizer

        if isinstance(config, str):
            config = {"from_pretrained": config}

        from_pretrained_name = config.get("from_pretrained", "bert-base-uncased")
        kwargs_dict = dict(kwargs, do_lower_case="uncased" in from_pretrained_name)
        self._tokenizer = BertTokenizer.from_pretrained(
            from_pretrained_name, **kwargs_dict
        )
        self._max_seq_length = config.get("max_seq_length", 70)
        self._probability = config.get("mask_probability", 0)
        self._corrupt_prob = config.get("corrupt_probability", 0)
        self._corrupt_caption_prob = config.get("corrupt_caption_probability", 0)

    def __call__(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        output = get_pair_text_tokens(item, self)
        output["text"] = output["tokens_masked"]
        output["tokens"] = output["tokens_masked"]

        if self._corrupt_prob > 0:
            contrastive_label, corrupt_output = self._get_contrastive_output(item)
            output["input_ids_corrupt"] = corrupt_output["input_ids"]
            output["segment_ids_corrupt"] = corrupt_output["segment_ids"]
            output["input_mask_corrupt"] = corrupt_output["input_mask"]
            output["contrastive_label"] = contrastive_label
        return output

    def _get_contrastive_output(
        self, item: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        error_msg = (
            "'{}' are required in the annotations for VinVL pretraining."
            + "These should be created using the MMF feature extraction script for MMF."
            + "To run finetuning or pretraining without contrastive loss,"
            + " set corrupt_probability to 0."
        )
        assert "random_captions" in item, error_msg.format("random_captions")
        assert "random_labels" in item, error_msg.format("random_labels")
        assert "text_b" in item, error_msg.format("text_b")

        corrupt_item = copy.deepcopy(item)
        p_match = 1 - self._corrupt_prob
        p_caption = self._corrupt_prob * self._corrupt_caption_prob
        p_label = self._corrupt_prob * (1 - self._corrupt_caption_prob)
        contrastive_label = torch.multinomial(
            torch.tensor([p_match, p_caption, p_label]), num_samples=1
        ).long()
        if contrastive_label == 0:
            pass
        elif contrastive_label == 1:
            num_subs = len(item["random_captions"])
            neg_index = torch.randint(num_subs, (1,))
            corrupt_item["text"] = item["random_captions"][neg_index]
        else:
            num_subs = len(item["random_labels"])
            neg_index = torch.randint(num_subs, (1,))
            corrupt_item["text_b"] = item["random_labels"][neg_index]

        return contrastive_label, get_pair_text_tokens(corrupt_item, self)

    def _token_transform(
        self, tokens: List[str], tokens_b: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, int, int, List[str]]:
        tokens = [self._CLS_TOKEN] + tokens + [self._SEP_TOKEN]
        if tokens_b:
            tokens += tokens_b + [self._SEP_TOKEN]

        input_ids = self._convert_tokens_to_ids(tokens)
        token_len = len(input_ids)
        token_pad = self._max_seq_length - token_len
        # Zero-pad up to the sequence length.
        input_ids += [self._PAD_TOKEN_ID] * token_pad
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        return input_ids_tensor, token_len, token_pad, tokens

    def _convert_to_indices(
        self,
        tokens_a: List[str],
        tokens_b: Optional[List[str]] = None,
        probability: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        """
        BERT encodes
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        """
        tokens_a_original = tokens_a.copy()
        tokens_a, label_a = self._random_word(tokens_a, probability=probability)
        segment_ids = [0] * (len(tokens_a) + 2)

        if tokens_b:
            tokens_b_original = tokens_b.copy()
            tokens_b, label_b = self._random_word(tokens_b, probability=probability)
            lm_label_ids = [-1] + label_a + [-1] + label_b + [-1]
            assert len(tokens_b) > 0
            segment_ids += [1] * (len(tokens_b) + 1)
        else:
            tokens_b_original = None
            lm_label_ids = [-1] + label_a + [-1]

        input_ids_masked, token_len, token_pad, tokens_masked = self._token_transform(
            tokens_a, tokens_b
        )
        input_ids_original, _, _, _ = self._token_transform(
            tokens_a_original, tokens_b_original
        )

        input_mask = [1] * token_len + [0] * token_pad
        segment_ids += [0] * token_pad
        lm_label_ids += [-1] * token_pad

        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        lm_label_ids = torch.tensor(lm_label_ids, dtype=torch.long)
        return {
            "input_ids_masked": input_ids_masked,  # specifically for MLM heads
            "input_ids": input_ids_original,
            # input_mask is non-padding (1) vs padding (0) mask (not MLM token masking)
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "lm_label_ids": lm_label_ids,
            "tokens_masked": tokens_masked,
        }
