from torch import nn


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()

    def register_vocab(self):
        raise NotImplementedError("register_vocab function not defined by this"
                                  "model. Register vocab function must be "
                                  " defined and should register vocabs for"
                                  " text and context.")

    @classmethod
    def init_args(cls, parser):
        return parser
