import torch

from pythia.common.registry import registry
from pythia.utils.vocab import Vocab
from pythia.utils.configuration import ConfigNode


class BaseProcessor:
    def __init__(self, name):
        self.name = name

    def __call__(self, config, info, *args, **kwargs):
        return info


@registry.register_processor("vocab")
class VocabProcessor(BaseProcessor):
    MAX_LENGTH_DEFAULT = 50
    def __init__(self, config, *args, **kwargs):
        self.vocab = Vocab(**config)
        self.writer = registry.get("writer")
        self.text_processor = None

        if hasattr(config, "max_length"):
            self.max_length = config.max_length
        else:
            self.writer.write("No 'max_length' parameter in VocabProcessor's"
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
            raise RuntimeError("Argument passed to vocab processor must be"
                               " a dict with either 'text' or 'tokens' as"
                               " keys")
        if "tokens" in item:
            tokens = item["tokens"]
            indices = self._map_strings_to_indices(item["tokens"])
        elif "text" in item:
            if self.text_processor is None:
                raise RuntimeError("If tokens are not provided, a text"
                                   " processor must be provided for vocab"
                                   " class")
            tokens = self.text_processor(item["text"])
            indices = self._map_strings_to_indices(tokens)
        else:
            raise RuntimeError("A dict with either 'text' or 'tokens' keys"
                               " must be passed to vocab processor")

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.new_full(self.max_length,
                                fill_value=self.vocab.get_pad_index(),
                                dtype=torch.int)

        for idx, token in tokens:
            output[idx] = self.vocab.stoi[token]

        return tokens
