from torch import nn

from pythia.core.registry import registry
from pythia.modules.embeddings import TextEmbedding
from pythia.modules.layers import Identity


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()

    def register_vocab(self):
        raise NotImplementedError("register_vocab function not defined by this"
                                  "model. Register vocab function must be "
                                  " defined and should register vocabs for"
                                  " text and context.")

    def _init_text_embedding(self, attr='text_embeddings',
                             bidirectional=False):
        text_embeddings = []
        text_embeddings_list_config = self.config[attr]

        self.embeddings_out_dim = 0

        text_vocab = registry.get("vocabs." + attr.split("_")[0] + "_vocab")

        if text_vocab.type == "model":
            # If vocab type is model, it is probably a fasttext model
            # which means we will get the embedding vectors directly
            # no need to do anything and just pass them through identity
            text_embeddings = nn.ModuleList([Identity()])
            setattr(self, attr + "_out_dim", text_vocab.get_dim())
            setattr(self, attr, text_embeddings)
            return

        for text_embedding in text_embeddings_list_config:
            embedding_type = text_embedding['type']
            embedding_kwargs = text_embedding['params']
            embedding_kwargs['bidirectional'] = bidirectional
            self._update_text_embedding_args(embedding_kwargs)

            embedding = TextEmbedding(text_vocab, embedding_type,
                                      **embedding_kwargs)
            text_embeddings.append(embedding)
            self.embeddings_out_dim += embedding.text_out_dim

        setattr(self, attr + "_out_dim", self.embeddings_out_dim)
        delattr(self, "embeddings_out_dim")
        setattr(self, attr, nn.ModuleList(text_embeddings))

    @classmethod
    def init_args(cls, parser):
        return parser
