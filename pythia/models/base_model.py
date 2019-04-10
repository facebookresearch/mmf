import collections

from torch import nn


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()

    def register_vocab(self):
        raise NotImplementedError("register_vocab function not defined by this"
                                  "model. Register vocab function must be "
                                  " defined and should register vocabs for"
                                  " text and context.")

    @classmethod
    def init_args(cls, parser):
        return parser

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)

        # Make sure theat the output from the model is a Mapping
        assert isinstance(result, collections.Mapping), "A dict must be \
            returned from forward of the model"
        return result
