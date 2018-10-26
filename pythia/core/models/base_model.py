from torch import nn

from pythia.core.registry import Registry
from pythia.modules.embeddings import TextEmbedding


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
        text_embeddings_list_config = self.config['text_embeddings']

        self.embeddings_out_dim = 0

        text_vocab = Registry.get('vocabs.text_vocab')

        for text_embedding in text_embeddings_list_config:
            embedding_type = text_embedding['type']
            embedding_kwargs = text_embedding['params']
            embedding_kwargs['bidirectional'] = bidirectional
            self._update_text_embedding_args(embedding_kwargs)

            embedding = TextEmbedding(text_vocab, embedding_type,
                                      **embedding_kwargs)
            text_embeddings.append(embedding)
            self.embeddings_out_dim += embedding.text_out_dim

        setattr(self, attr + '_out_dim', self.embeddings_out_dim)
        delattr(self, 'embeddings_out_dim')
        setattr(self, attr, nn.ModuleList(text_embeddings))
