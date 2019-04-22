# Copyright (c) Facebook, Inc. and its affiliates.
import os
import warnings
from collections import Counter

import torch

from pythia.common.registry import registry
from pythia.utils.configuration import ConfigNode
from pythia.utils.distributed_utils import is_main_process, synchronize
from pythia.utils.general import get_pythia_root
from pythia.utils.text_utils import VocabDict
from pythia.utils.vocab import Vocab, WordToVectorDict


class BaseProcessor:
    def __init__(self, config, *args, **kwargs):
        return

    def __call__(self, item, *args, **kwargs):
        return item


class Processor:
    def __init__(self, config, *args, **kwargs):
        self.writer = registry.get("writer")

        if not hasattr(config, "type"):
            raise AttributeError(
                "Config must have 'type' attribute to " "specify type of processor"
            )

        processor_class = registry.get_processor_class(config.type)

        params = {}
        if not hasattr(config, "params"):
            self.writer.write(
                "Config doesn't have 'params' attribute to "
                "specify parameters of the processor "
                "of type {}. Setting to default \{\}".format(config.type)
            )
        else:
            params = config.params

        self.processor = processor_class(params, *args, **kwargs)

        self._dir_representation = dir(self)

    def __call__(self, item):
        return self.processor(item)

    def __getattr__(self, name):
        if name in self._dir_representation:
            return getattr(self, name)
        elif hasattr(self.processor, name):
            return getattr(self.processor, name)
        else:
            raise AttributeError(name)


@registry.register_processor("vocab")
class VocabProcessor(BaseProcessor):
    MAX_LENGTH_DEFAULT = 50
    PAD_TOKEN = "<pad>"
    PAD_INDEX = 0

    def __init__(self, config, *args, **kwargs):
        if not hasattr(config, "vocab"):
            raise AttributeError(
                "config passed to the processor has no " "attribute vocab"
            )

        self.vocab = Vocab(*args, **config.vocab, **kwargs)
        self._init_extras(config)

    def _init_extras(self, config, *args, **kwargs):
        self.writer = registry.get("writer")
        self.preprocessor = None

        if hasattr(config, "max_length"):
            self.max_length = config.max_length
        else:
            warnings.warn(
                "No 'max_length' parameter in Processor's "
                "configuration. Setting to {}.".format(self.MAX_LENGTH_DEFAULT)
            )
            self.max_length = self.MAX_LENGTH_DEFAULT

        if hasattr(config, "preprocessor"):
            self.preprocessor = Processor(config.preprocessor, *args, **kwargs)

            if self.preprocessor is None:
                raise ValueError(
                    "No text processor named {} is defined.".format(config.preprocessor)
                )

    def __call__(self, item):
        indices = None
        if not isinstance(item, dict):
            raise TypeError(
                "Argument passed to the processor must be "
                "a dict with either 'text' or 'tokens' as "
                "keys"
            )
        if "tokens" in item:
            tokens = item["tokens"]
            indices = self._map_strings_to_indices(item["tokens"])
        elif "text" in item:
            if self.preprocessor is None:
                raise AssertionError(
                    "If tokens are not provided, a text "
                    "processor must be defined in the config"
                )

            tokens = self.preprocessor({"text": item["text"]})["text"]
            indices = self._map_strings_to_indices(tokens)
        else:
            raise AssertionError(
                "A dict with either 'text' or 'tokens' keys "
                "must be passed to the processor"
            )

        tokens, length = self._pad_tokens(tokens)

        return {"text": indices, "tokens": tokens, "length": length}

    def _pad_tokens(self, tokens):
        padded_tokens = [self.PAD_TOKEN] * self.max_length
        token_length = min(len(tokens), self.max_length)
        padded_tokens[:token_length] = tokens[:token_length]
        token_length = torch.tensor(token_length, dtype=torch.long)
        return padded_tokens, token_length

    def get_pad_index(self):
        return self.vocab.get_pad_index()

    def get_vocab_size(self):
        return self.vocab.get_size()

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.zeros(self.max_length, dtype=torch.long)
        output.fill_(self.vocab.get_pad_index())

        for idx, token in enumerate(tokens):
            output[idx] = self.vocab.stoi[token]

        return output


@registry.register_processor("glove")
class GloVeProcessor(VocabProcessor):
    def __init__(self, config, *args, **kwargs):
        if not hasattr(config, "vocab"):
            raise AttributeError(
                "Config passed to the processor has no " "attribute vocab"
            )
        vocab_processor_config = ConfigNode(config)
        # GloVeProcessor needs vocab type to be "intersected"
        vocab_processor_config.vocab.type = "intersected"

        if "vocab_file" not in vocab_processor_config.vocab:
            warnings.warn(
                "'vocab_file' key is not present in the config."
                " Switching to pretrained vocab."
            )

            vocab_processor_config.vocab.type = "pretrained"

        super().__init__(vocab_processor_config, *args, **kwargs)

    def __call__(self, item):
        indices = super().__call__(item)["text"]
        embeddings = torch.zeros(
            (len(indices), self.vocab.get_embedding_dim()), dtype=torch.float
        )

        for idx, index in enumerate(indices):
            embeddings[idx] = self.vocab.vectors[index]

        return {"text": embeddings}


@registry.register_processor("fasttext")
class FastTextProcessor(VocabProcessor):
    def __init__(self, config, *args, **kwargs):
        self._init_extras(config)

        needs_download = False
        model_file = config.model_file
        model_file = os.path.join(get_pythia_root(), config.model_file)

        if not hasattr(config, "model_file"):
            warnings.warn(
                "'model_file' key is required but missing "
                "from FastTextProcessor's config."
            )
            needs_download = True
        elif not os.path.exists(model_file):
            warnings.warn("No model file present at {}.".format(model_file))
            needs_download = True

        if needs_download:
            self.writer.write("Downloading FastText vectors", "info")
            model_file = self._download_model()

        synchronize()

        self._load_fasttext_model(model_file)

    def _download_model(self):
        model_file_path = os.path.join(
            get_pythia_root(), ".vector_cache", "wiki.en.bin"
        )

        if not is_main_process():
            return model_file_path

        if os.path.exists(model_file_path):
            self.writer.write(
                "Vectors already present at {}.".format(model_file_path), "info"
            )
            return model_file_path

        import torchtext

        torchtext.vocab.FastText("en", cache=os.path.dirname(model_file_path))

        self.writer.write("Vectors downloaded at {}.".format(model_file_path), "info")

        return model_file_path

    def _load_fasttext_model(self, model_file):
        from fastText import load_model

        self.writer.write("Loading fasttext model now from %s" % model_file)

        self.model = load_model(model_file)
        # String to Vector
        self.stov = WordToVectorDict(self.model)
        self.writer.write("Finished loading fasttext model")

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.full(
            (self.max_length, self.model.get_dimension()),
            fill_value=self.PAD_INDEX,
            dtype=torch.float,
        )

        for idx, token in enumerate(tokens):
            output[idx] = torch.from_numpy(self.stov[token])

        return output


@registry.register_processor("vqa_answer")
class VQAAnswerProcessor(BaseProcessor):
    DEFAULT_NUM_ANSWERS = 10

    def __init__(self, config, *args, **kwargs):
        self.writer = registry.get("writer")
        if not hasattr(config, "vocab_file"):
            raise AttributeError(
                "'vocab_file' argument required, but not "
                "present in AnswerProcessor's config"
            )

        self.answer_vocab = VocabDict(config.vocab_file, *args, **kwargs)

        self.preprocessor = None

        if hasattr(config, "preprocessor"):
            self.preprocessor = Processor(config.preprocessor)

            if self.preprocessor is None:
                raise ValueError(
                    "No processor named {} is defined.".format(config.preprocessor)
                )

        if hasattr(config, "num_answers"):
            self.num_answers = config.num_answers
        else:
            self.num_answers = self.DEFAULT_NUM_ANSWERS
            warnings.warn(
                "'num_answers' not defined in the config. "
                "Setting to default of {}".format(self.DEFAULT_NUM_ANSWERS)
            )

    def __call__(self, item):
        tokens = None

        if not isinstance(item, dict):
            raise TypeError("'item' passed to processor must be a dict")

        if "answer_tokens" in item:
            tokens = item["answer_tokens"]
        elif "answers" in item:
            if self.preprocessor is None:
                raise AssertionError(
                    "'preprocessor' must be defined if you "
                    "don't pass 'answer_tokens'"
                )

            tokens = [
                self.preprocessor({"text": answer})["text"]
                for answer in item["answers"]
            ]
        else:
            raise AssertionError(
                "'answers' or 'answer_tokens' must be passed"
                " to answer processor in a dict"
            )

        answers_indices = torch.zeros(self.num_answers, dtype=torch.int)
        answers_indices.fill_(self.answer_vocab.get_unk_index())

        for idx, token in enumerate(tokens):
            answers_indices[idx] = self.answer_vocab.word2idx(token)

        answers_scores = self.compute_answers_scores(answers_indices)

        return {
            "answers": tokens,
            "answers_indices": answers_indices,
            "answers_scores": answers_scores,
        }

    def get_vocab_size(self):
        return self.answer_vocab.num_vocab

    def get_true_vocab_size(self):
        return self.answer_vocab.num_vocab

    def word2idx(self, word):
        return self.answer_vocab.word2idx(word)

    def idx2word(self, idx):
        return self.answer_vocab.idx2word(idx)

    def compute_answers_scores(self, answers_indices):
        scores = torch.zeros(self.get_vocab_size(), dtype=torch.float)
        gt_answers = list(enumerate(answers_indices))
        unique_answers = set(answers_indices.tolist())

        for answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]

                matching_answers = [item for item in other_answers if item[1] == answer]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            avg_acc = sum(accs) / len(accs)

            if answer != self.answer_vocab.UNK_INDEX:
                scores[answer] = avg_acc

        return scores


@registry.register_processor("soft_copy_answer")
class SoftCopyAnswerProcessor(VQAAnswerProcessor):
    DEFAULT_MAX_LENGTH = 50

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        if not hasattr(config, "use_soft_copy"):
            warnings.warn(
                "SoftCopyAnswerProcessor's config doesn't have field"
                " 'use_soft_copy'. Setting to default of False"
            )
            self.use_soft_copy = False
        else:
            self.use_soft_copy = config.use_soft_copy

        if hasattr(config, "max_length"):
            self.max_length = config.max_length
        else:
            self.max_length = self.DEFAULT_MAX_LENGTH
            warnings.warn(
                "'max_length' not defined in the config. "
                "Setting to default of {}".format(self.DEFAULT_MAX_LENGTH)
            )

        self.context_preprocessor = None
        if hasattr(config, "context_preprocessor"):
            self.context_preprocessor = Processor(config.context_preprocessor)

    def get_vocab_size(self):
        answer_vocab_nums = self.answer_vocab.num_vocab

        if self.use_soft_copy is True:
            answer_vocab_nums += self.max_length

        return answer_vocab_nums

    def get_true_vocab_size(self):
        return self.answer_vocab.num_vocab

    def __call__(self, item):
        answers = item["answers"]
        scores = super().__call__({"answers": answers})

        if self.use_soft_copy is False:
            return scores

        indices = scores["answers_indices"]
        answers = scores["answers"]
        scores = scores["answers_scores"]

        tokens_scores = scores.new_zeros(self.max_length)
        tokens = item["tokens"]
        length = min(len(tokens), self.max_length)

        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)

        if self.context_preprocessor is not None:
            tokens = [
                self.context_preprocessor({"text": token})["text"] for token in tokens
            ]

        answer_counter = Counter(answers)

        for idx, token in enumerate(tokens[:length]):
            if answer_counter[token] == 0:
                continue
            accs = []

            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == token]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)

            tokens_scores[idx] = sum(accs) / len(accs)

        # Scores are already proper size, see L314. Now,
        # fix scores for soft copy candidates
        scores[-len(tokens_scores) :] = tokens_scores
        return {
            "answers": answers,
            "answers_indices": indices,
            "answers_scores": scores,
        }


@registry.register_processor("simple_word")
class SimpleWordProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        from pythia.utils.text_utils import word_tokenize

        self.tokenizer = word_tokenize

    def __call__(self, item):
        return {"text": self.tokenizer(item["text"])}


@registry.register_processor("simple_sentence")
class SimpleSentenceProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        from pythia.utils.text_utils import tokenize

        self.tokenizer = tokenize

    def __call__(self, item):
        return {"text": self.tokenizer(item["text"])}


@registry.register_processor("bbox")
class BBoxProcessor(VocabProcessor):
    def __init__(self, config, *args, **kwargs):
        from pythia.utils.dataset_utils import build_bbox_tensors

        self.lambda_fn = build_bbox_tensors
        self._init_extras(config)

    def __call__(self, item):
        info = item["info"]
        if self.preprocessor is not None:
            info = self.preprocessor(info)

        return {"bbox": self.lambda_fn(info)}
