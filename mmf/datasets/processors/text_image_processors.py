# Copyright (c) Facebook, Inc. and its affiliates.

from math import nan

import numpy as np
import torch
from mmf.common.registry import registry
from mmf.datasets.processors.processors import BaseProcessor
from mmf.utils.process_text_image import text_token_overlap_with_bbox
from mmf.utils.text import is_punctuation
from scipy import interpolate
from transformers.tokenization_auto import AutoTokenizer


class TextImageProcessor(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        text_tokenizer_config = config.text_processor.tokenizer_config
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer_config.type, **text_tokenizer_config.params
        )

    def sample_traces(self, utterances, utterance_times, traces):
        tokens, index_tokens, token_weights, token_traces = [], [], [], []
        for idx, utterance in enumerate(utterances):
            if self.config.remove_punctuations:
                utterance = self.remove_punctuation(utterance)

            tokens_from_utterance = self.text_tokenizer.tokenize(utterance)
            indexed_tokens_from_utterance = self.text_tokenizer.convert_tokens_to_ids(
                tokens_from_utterance
            )
            (
                token_times,
                token_weights_from_utterance,
            ) = self.determine_token_weight_and_times(
                tokens_from_utterance,
                utterance_times[idx][0].item(),
                utterance_times[idx][1].item(),
            )
            token_traces_from_utterance = self.traces_from_token(
                token_times, traces, self.config.trace_num_samples
            )

            tokens += tokens_from_utterance
            index_tokens += indexed_tokens_from_utterance
            token_weights += token_weights_from_utterance
            token_traces += token_traces_from_utterance
        return (
            tokens,
            torch.IntTensor(index_tokens),
            torch.FloatTensor(token_weights),
            torch.FloatTensor(token_traces),
        )

    def determine_token_weight_and_times(self, tokens, start, end):
        return (
            torch.FloatTensor([start, end]).repeat(len(tokens), 1),
            torch.FloatTensor([1 / len(tokens)]).repeat(len(tokens)),
        )

    def traces_from_token(self, token_times, traces, num_samples):
        # print(f"token_times: {token_times}")
        assert num_samples >= 2, "Must sample at least the start and end traces."
        sampled_traces = []
        np_traces = np.array(traces)
        tracesx, tracesy, tracest = np_traces[:, 0], np_traces[:, 1], np_traces[:, 2]
        for start, end in token_times:
            interval = (end - start) / (num_samples - 1)
            times = [start + i * interval for i in range(num_samples)]
            sample_trace_x = np.array(
                [
                    interpolate.interp1d(
                        tracest,
                        tracesx,
                        bounds_error=False,
                        fill_value=traces[0][0].item()
                        if t < traces[0][2]
                        else traces[-1][0].item()
                        if t > traces[-1][2]
                        else nan,
                    )(t)
                    for t in times
                ]
            )[..., None]
            sample_trace_y = np.array(
                [
                    interpolate.interp1d(
                        tracest,
                        tracesy,
                        bounds_error=False,
                        fill_value=traces[0][1].item()
                        if t < traces[0][2]
                        else traces[-1][1].item()
                        if t > traces[-1][2]
                        else nan,
                    )(t)
                    for t in times
                ]
            )[..., None]
            trace_per_utter = np.stack((sample_trace_x, sample_trace_y), axis=1)
            sampled_traces.append(trace_per_utter)
        return sampled_traces

    def remove_punctuation(self, word):
        chars = ""
        for char in word:
            if not is_punctuation(char):
                chars += char
        return chars


@registry.register_processor("text_token_overlap_with_image_bbox")
class TextTokenOverlapWithImageBboxProcessor(TextImageProcessor):
    def __call__(self, item):
        utterances = item["utterances"]
        utterance_times = item["utterance_times"]
        traces = item["traces"]
        image_info = item["image_info"]
        tokens, index_tokens, token_weights, token_traces = self.sample_traces(
            utterances, utterance_times, traces
        )
        percent_overlaps = text_token_overlap_with_bbox(
            token_traces, image_info["bbox"], self.config.trace_num_samples,
        )
        max_seq_length = self.config.text_processor.max_seq_length

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            index_tokens = index_tokens[:max_seq_length]
            token_weights = token_weights[:max_seq_length]
            percent_overlaps = percent_overlaps[:max_seq_length]
        else:
            pads = [0] * (max_seq_length - len(tokens))
            tokens += pads
            index_tokens = torch.cat((index_tokens, torch.tensor(pads)))
            token_weights = torch.cat((token_weights, torch.tensor(pads)))
            num_bboxes = percent_overlaps.shape[-1]
            percent_overlaps = torch.cat(
                (
                    percent_overlaps,
                    torch.tensor(pads).unsqueeze(1).repeat(1, num_bboxes),
                )
            )

        return {
            "tokens": tokens,
            "input_ids": index_tokens,
            "token_weights": token_weights,
            "percent_overlaps": percent_overlaps,
        }


@registry.register_processor("text_token_closeness_with_image_bbox")
class TextTokenClosenessWithImageBboxProcessor(TextImageProcessor):
    pass
