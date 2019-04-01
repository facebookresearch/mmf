import torch

from pythia.common.registry import registry
from pythia.utils.vocab import Vocab, WordToVectorDict
from pythia.utils.configuration import ConfigNode
from pythia.utils.text_utils import VocabDict


class BaseProcessor:
    def __init__(self, config, *args, **kwargs):
        return

    def __call__(self, item, *args, **kwargs):
        return item

class Processor:
    def __init__(self, config):
        self.writer = registry.get("writer")

        if not hasattr(config, "type"):
            raise AttributeError("Config must have 'type' attribute to "
                                 "specify type of processor")

        processor_class = registry.get_processor_class(config.type)

        params = {}
        if not hasattr(config, "params"):
            self.writer.write("Config doesn't have 'params' attribute to "
                              "specify parameters of the processor "
                              "of type {}. Setting to default \{\}"
                              .format(config.type))
            params = config.params
        self.processor = processor_class(params)

    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)
        elif hasattr(self.processor, name):
            return getattr(self.processor, name)
        else:
            raise AttributeError(name)

@registry.register_processor("vocab")
class VocabProcessor(BaseProcessor):
    MAX_LENGTH_DEFAULT = 50
    def __init__(self, config, *args, **kwargs):
        self.vocab = Vocab(**config)
        self._init_extras(config)

    def _init_extras(self, config):
        self.writer = registry.get("writer")
        self.text_processor = None

        if hasattr(config, "max_length"):
            self.max_length = config.max_length
        else:
            self.writer.write("No 'max_length' parameter in Processor's"
                              " configuration. Setting to {}."
                              .format(self.MAX_LENGTH_DEFAULT),
                              "warning")
            self.max_length = self.MAX_LENGTH_DEFAULT

        if hasattr(config, "text_processor"):
            self.text_processor = registry.get_processor_class(
                config.text_processor
            )

            if self.text_processor is None:
                raise RuntimeError("No text processor named {} is defined."
                                   .format(config.text_processor))

    def __call__(self, item):
        indices = None
        if not isinstance(item, dict):
            raise RuntimeError("Argument passed to the processor must be"
                               " a dict with either 'text' or 'tokens' as"
                               " keys")
        if "tokens" in item:
            tokens = item["tokens"]
            indices = self._map_strings_to_indices(item["tokens"])
        elif "text" in item:
            if self.text_processor is None:
                raise RuntimeError("If tokens are not provided, a text"
                                   " processor must be defined in"
                                   " the config")
            tokens = self.text_processor({"text": item["text"]})["text"]
            indices = self._map_strings_to_indices(tokens)
        else:
            raise RuntimeError("A dict with either 'text' or 'tokens' keys"
                               " must be passed to the processor")

        return {
            "text": indices
        }

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.full(self.max_length,
                             fill_value=self.vocab.get_pad_index(),
                                dtype=torch.int)

        for idx, token in tokens:
            output[idx] = self.vocab.stoi[token]

        return output


@regsitry.register_processor("glove")
class GloVeProcessor(VocabProcessor):
    def __init__(self, config, *args, **kwargs):
        vocab_processor_config = ConfigNode(config)
        # GloVeProcessor needs vocab type to be "intersected"
        vocab_processor_config.type = "intersected"

        if "vocab_file" not in vocab_processor_config:
            registry.get("writer").write("'vocab_file' key is not present in"
                                         " the config. Switching to"
                                          " pretrained vocab.", "warning")
            vocab_processor_config.type = "pretrained"

        super().__init__(vocab_processor_config)

    def __call__(self, item):
        indices = super().__call__(item)
        embeddings = torch.zeros((len(indices),
                                 self.vocab.get_embedding_dim()),
                                 dtype=torch.float)

        for idx, index in enumerate(indices):
            embeddings[idx] = self.vocab.vectors[index]

        return {
            "text": embeddings
        }

@registry.register_processor("fasttext")
class FastTextProcessor(VocabProcessor):
    def __init__(self, config, *args, **kwargs):
        self._init_extras(config)

        if not hasattr(config, "model_file"):
            raise AttributeError("'model_file' key is required but missing from"
                                 " FastTextProcessor's config.")

        self._load_fasttext_model(config.model_file)

    def _load_fasttext_model(self, model_file):
        from fastText import load_model

        self.writer.write("Loading fasttext model now from %s" % model_file)

        self.model = load_model(model_file)
        # String to Vector
        self.stov = WordToVectorDict(self.model)

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.full((self.max_length, self.model.get_dimension()),
                            fill_value=self.vocab.get_pad_index(),
                            dtype=torch.float)

        for idx, token in tokens:
            output[idx] = self.stov[token]

        return output


@registry.register_processor("vqa_answer")
class VQAAnswerProcessor(BaseProcessor):
    DEFAULT_NUM_ANSWERS = 10
    def __init__(self, config):
        self.write = registry.get("writer")
        if not hasattr(config, 'vocab_file'):
            raise AttributeError("'vocab_file' argument required, but not "
                                 "present in AnswerProcessor's config")
        self.answer_vocab = VocabDict(config.vocab_file)

        self.text_processor = None

        if hasattr(config, "text_processor"):
            self.text_processor = registry.get_processor_class(
                config.text_processor
            )

            if self.text_processor is None:
                raise RuntimeError("No text processor named {} is defined."
                                   .format(config.text_processor))

        if hasattr(config, "num_answers"):
            self.num_answers = config.num_answers
        else:
            self.writer.write("'num_answers' not defined in the config. "
                              "Setting to default of {}"
                              .format(self.DEFAULT_NUM_ANSWERS), "warning")

    def __call__(self, item):
        tokens = None

        if not isinstance(item, dict):
            raise RuntimeError("'item' passed to processor must be a dict")

        if hasattr(item, 'answer_tokens'):
            tokens = answer_tokens
        elif hasattr(item, 'answers'):
            if self.text_processor is None:
                raise RuntimeError("'text_processor' must be defined if you "
                                   "don't pass 'answer_tokens'")

            tokens = [self.text_processor({'text': answer})["text"]
                      for answer in item['answers']]
        else:
            raise RuntimeError("'answers' or 'answer_tokens' must be passed"
                               " to answer processor in a dict")

        answers_indices = torch.full(self.num_answers, fill_value=-1,
                                     dtype=torch.int)

        for idx, token in enumerate(tokens):
            answers_indices[idx] = self.answer_dict.word2idx(token)

        answers_scores = self.compute_answers_scores(answers_indices)

        return {
            "answers": answer_scores,
            "answers_scores": answers_scores
        }

    def get_vocab_size(self):
        return self.answer_dict.num_vocab

    def compute_answers_scores(self, answers_indices):
        scores = torch.zeros(self.get_vocab_size(), dtype=torch.float)
        gt_answers = list(enumerate(answer_indices))
        unique_answers = set(answers_indices.tolist())

        for answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers
                                 if item != gt_answer]

                matching_answers = [item for item in other_answers
                                    if item[1] == answer]
                acc = min(1, float(len(matching_ans)) / 3)
                accs.append(acc)
            avg_acc = torch.mean(accs)

            if answer == self.answer_dict.UNK_INDEX:
                scores[answer] = 0
            else:
                scores[answer] = avg_acc

        return scores


@registry.register_class("simple_word")
class SimpleWordProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        from pythia.utils.text_utils import word_tokenize
        self.tokenizer = word_tokenize

    def __call__(self, item):
        return {"text": self.tokenizer(item["text"])}


@registry.register_class("simple_sentence")
class SimpleSentenceProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        from pythia.utils.text_utils import tokenize
        self.tokenizer = tokenize

    def __call__(self, item):
        return {"text": self.tokenizer(item["text"])}
