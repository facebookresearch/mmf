import os
import sys
import torch
import numpy as np

from collections import defaultdict
from torchtext import vocab

EMBEDDING_NAME_CLASS_MAPPING = {
    'glove': 'GloVe',
    'fasttext': 'FastText'
}


class Vocab:
    @classmethod
    def get(cls, **params):
        vocab_type = params.get('type', 'pretrained')
        vocab_params = {}
        vocab_params.update(params)
        vocab_params.pop('type')
        # Stores final parameters extracted from vocab_params

        if vocab_type == 'random':
            if vocab_params['vocab_file'] is None:
                print("No vocab path passed for vocab")
                sys.exit(0)
            return BaseVocab(**vocab_params)

        elif vocab_type == 'custom':
            if vocab_params['vocab_file'] is None or \
               vocab_params['embedding_file'] is None:
                print("No vocab path or embedding_file passed for vocab")
                sys.exit(0)
            return CustomVocab(**vocab_params)

        elif vocab_type == 'pretrained':
            return PretrainedVocab(**vocab_params)

        elif vocab_type == 'intersected':
            if vocab_params['vocab_file'] is None or \
               vocab_params['embedding_name'] is None:
                print("No vocab path or embedding_name passed for vocab")
                sys.exit(0)

            return IntersectedVocab(**vocab_params)
        elif vocab_type == 'model':
            if vocab_params['name'] == 'fasttext':
                return ModelVocab(**vocab_params)
        else:
            raise RuntimeError("Unknown vocab type: %s" % vocab_type)


class BaseVocab:
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'
    UNK_TOKEN = '<unk>'

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, vocab_file=None, embedding_dim=300):
        """Vocab class to be used when you want to train word embeddings from
        scratch based on a custom vocab. This will initialize the random
        vectors for the vocabulary you pass. Get the vectors using
        `get_vectors` function. This will also create random embeddings for
        some predefined words like PAD - <pad>, SOS - <s>, EOS - </s>,
        UNK - <unk>.

        Parameters
        ----------
        vocab_file : str
            Path of the vocabulary file containing one word per line
        embedding_dim : int
            Size of the embedding

        """
        self.type = "base"
        self.word_dict = {}
        self.itos = {}

        self.itos[self.PAD_INDEX] = self.PAD_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN
        self.itos[self.UNK_INDEX] = self.UNK_TOKEN

        self.word_dict[self.SOS_TOKEN] = self.SOS_INDEX
        self.word_dict[self.EOS_TOKEN] = self.EOS_INDEX
        self.word_dict[self.PAD_TOKEN] = self.PAD_INDEX
        self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX

        index = len(self.itos.keys())

        self.total_predefined = len(self.itos.keys())

        if vocab_file is not None:

            if not os.path.exists(vocab_file):
                print("Vocab not found at " + vocab_file)
                sys.exit(1)

            with open(vocab_file, 'r') as f:
                for line in f:
                    self.itos[index] = line.strip()
                    self.word_dict[line.strip()] = index
                    index += 1

        self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX
        # Return unk index by default
        self.stoi = defaultdict(lambda: self.UNK_INDEX)
        self.stoi.update(self.word_dict)

        self.vectors = torch.FloatTensor(self.get_size(), embedding_dim)

    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi

    def get_size(self):
        return len(self.itos)

    def get_vectors(self):
        return getattr(self, 'vectors', None)

    def get_embedding(self, cls, **embedding_kwargs):
        vector_dim = len(self.vectors[0])
        embedding_kwargs['vocab_size'] = self.get_size()

        embedding_dim = embedding_kwargs['embedding_dim']
        embedding_kwargs['embedding_dim'] = vector_dim

        embedding = None

        if cls == torch.nn.Embedding:
            embedding = torch.nn.Embedding(self.get_size(), vector_dim)
        else:
            embedding = cls(**embedding_kwargs)

        if hasattr(embedding, 'embedding'):
            embedding.embedding = torch.nn.Embedding.from_pretrained(
                self.vectors, freeze=False
            )
        else:
            embedding = torch.nn.Embedding.from_pretrained(self.vectors,
                                                           freeze=False)

        if vector_dim == embedding_dim:
            return embedding
        else:
            return torch.nn.Sequential([
                embedding,
                torch.nn.Linear(vector_dim, embedding_dim)
            ])


class CustomVocab(BaseVocab):
    def __init__(self, vocab_file, embedding_file, data_dir=None):
        """Use this vocab class when you have a custom vocab as well as a
        custom embeddings file.

        This will inherit vocab class, so you will get predefined tokens with
        this one.

        IMPORTANT: To init your embedding, get your vectors from this class's
        object by calling `get_vectors` function

        Parameters
        ----------
        vocab_file : str
            Path of custom vocabulary
        embedding_file : str
            Path to custom embedding inititalization file
        data_dir : str
            Path to data directory if embedding file is not an absolute path.
            Default: None
        """
        super(CustomVocab, self).__init__(vocab_file)
        self.type = "custom"

        if not os.path.isabs(embedding_file) and data_dir is not None:
            embedding_file = os.path.join(data_dir, embedding_file)

        if not os.path.exists(embedding_file):
            from pythia.core.registry import Registry
            writer = Registry.get('writer')
            error = "Embedding file path %s doesn't exist" % embedding_file
            if writer is not None:
                writer.write(error, "error")
                sys.exit(0)
            else:
                raise RuntimeError(error)

        embedding_vectors = torch.from_numpy(np.load(embedding_file))

        self.vectors = torch.FloatTensor(self.get_size(),
                                         len(embedding_vectors[0]))

        for i in range(0, 4):
            self.vectors[i] = torch.ones_like(self.vectors[i]) * 0.1 * i

        for i in range(4, self.get_size()):
            self.vectors[i] = embedding_vectors[i - 4]


class IntersectedVocab(BaseVocab):
    def __init__(self, vocab_file, embedding_name):
        """Use this vocab class when you have a custom vocabulary class but you
        want to use pretrained embedding vectos for it. This will only load
        the vectors which intersect with your vocabulary. Use the
        embedding_name specified in torchtext's pretrained aliases:
        ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d',
         'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d',
         'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
         'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d',
         'glove.6B.200d', 'glove.6B.300d']

        Parameters
        ----------
        vocab_file : str
            Vocabulary file containing list of words with one word per line
            which will be used to collect vectors
        embedding_name : str
            Embedding name picked up from the list of the pretrained aliases
            mentioned above
        """
        super(IntersectedVocab, self).__init__(vocab_file)

        self.type = "intersected"

        name = embedding_name.split('.')[0]
        dim = embedding_name.split('.')[2][:-1]
        middle = embedding_name.split('.')[1]

        class_name = EMBEDDING_NAME_CLASS_MAPPING[name]

        if not hasattr(vocab, class_name):
            from pythia.core.registry import Registry
            writer = Registry.get('writer')
            error = "Unknown embedding type: %s" % name, "error"
            if writer is not None:
                writer.write(error, "error")
                sys.exit(0)
            else:
                raise RuntimeError(error)

        params = [middle]

        if name == 'glove':
            params.append(int(dim))

        embedding = getattr(vocab, class_name)(*params)

        self.vectors = torch.FloatTensor(self.get_size(),
                                         len(embedding.vectors[0]))

        for i in range(0, 4):
            self.vectors[i] = torch.ones_like(self.vectors[i]) * 0.1 * i

        for i in range(4, self.get_size()):
            word = self.itos[i]
            embedding_index = embedding.stoi.get(word, None)

            if embedding_index is None:
                self.vectors[i] = self.vectors[self.UNK_INDEX].clone()
            else:
                self.vectors[i] = embedding.vectors[embedding_index]


class PretrainedVocab(BaseVocab):
    def __init__(self, embedding_name):
        """Use this if you want to use pretrained embedding. See description
        of IntersectedVocab to get a list of the embedding available from
        torchtext

        Parameters
        ----------
        embedding_name : str
            Name of the pretrained alias for the embedding to used
        """
        self.type = "pretrained"

        if embedding_name not in vocab.pretrained_aliases:
            from pythia.core.registry import Registry
            writer = Registry.get('writer')
            error = "Unknown embedding type: %s" % embedding_name, "error"
            if writer is not None:
                writer.write(error, "error")
                sys.exit(0)
            else:
                raise RuntimeError(error)
        embedding = vocab.pretrained_aliases[embedding_name]()

        self.UNK_INDEX = 3
        self.stoi = defaultdict(lambda: self.UNK_INDEX)
        self.itos = {}

        self.itos[self.PAD_INDEX] = self.PAD_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN
        self.itos[self.UNK_INDEX] = self.UNK_TOKEN

        self.stoi[self.SOS_TOKEN] = self.SOS_INDEX
        self.stoi[self.EOS_TOKEN] = self.EOS_INDEX
        self.stoi[self.PAD_TOKEN] = self.PAD_INDEX
        self.stoi[self.UNK_TOKEN] = self.UNK_INDEX

        self.vectors = torch.FloatTensor(len(self.itos.keys())
                                         + len(embedding.itos),
                                         len(embedding.vectors[0]))

        for i in range(4):
            self.vectors[i] = torch.ones_like(self.vectors[i]) * 0.1 * i

        index = 4
        for word in embedding.stoi:
            self.itos[index] = word
            self.stoi[word] = index
            actual_index = embedding.stoi[word]
            self.vectors[index] = embedding.vectors[actual_index]
            index += 1


class WordToVectorDict:
    def __init__(self, model):
        self.model = model

    def __getitem__(self, word):
        return self.model.get_word_vector(word)


class ModelVocab(BaseVocab):
    def __init__(self, name, model_file):
        """Special vocab which is not really vocabulary but instead a model
        which returns embedding directly instead of vocabulary. This is just
        an abstraction over a model which generates embeddings directly.
        For e.g. for fasttext model we encapsulate it inside this and provide
        it as a vocab so that the API of the vocab remains same.

        NOTE: stoi's functionality will remain same but it is actually calling
        a function to get word vectors. Currently, only fasttext is supported.

        Parameters
        ----------
        name : str
            Name of the embedding model which this vocab currently is loading
        model_file : str
            File from which model will be loaded. This API might need to be
            changed in future.
        """
        super(ModelVocab, self).__init__()
        self.type = "model"
        if name != 'fasttext':
            raise RuntimeError("Model vocab only supports fasttext as of now")
        else:
            self._load_fasttext_model(model_file)

    def _load_fasttext_model(self, model_file):
        from fastText import load_model
        from pythia.core.registry import Registry

        Registry.get('writer').write("Loading fasttext model now from %s"
                                     % model_file)

        self.model = load_model(model_file)
        self.stoi = WordToVectorDict(self.model)

    def get_dim(self):
        return self.model.get_dimension()
