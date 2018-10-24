import os
import sys
import torch
import numpy as np

from collections import defaultdict
from torchtext import vocab

EMBEDDING_NAME_CLASS_MAPPING = {
    'embedding': 'GloVe',
    'fasttext': 'FastText'
}


class Vocab:
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'
    UNK_TOKEN = '<unk>'

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = None

    def __init__(self, vocab_path, embedding_dim=300):
        """Vocab class to be used when you want to train word embeddings from
        scratch based on a custom vocab. This will initialize the random
        vectors for the vocabulary you pass. Get the vectors using
        `get_vectors` function. This will also create random embeddings for
        some predefined words like PAD - <pad>, SOS - <s>, EOS - </s>,
        UNK - <unk>.

        Parameters
        ----------
        vocab_path : str
            Path of the vocabulary file containing one word per line
        embedding_dim : int
            Size of the embedding

        """
        if not os.path.exists(vocab_path):
            print("Vocab not found at " + vocab_path)
            sys.exit(1)

        self.word_dict = {}
        self.itos = {}

        self.itos[self.PAD_INDEX] = self.PAD_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN

        self.word_dict[self.SOS_TOKEN] = self.SOS_INDEX
        self.word_dict[self.EOS_TOKEN] = self.EOS_INDEX
        self.word_dict[self.PAD_INDEX] = self.PAD_INDEX

        index = len(self.itos.keys()) + 1

        total_predefined = len(self.itos.keys())

        with open(vocab_path, 'r') as f:
            for line in f:
                self.itos[index] = line.strip()
                self.word_dict[line.strip()] = index
                index += 1

        self.UNK_INDEX = self.word_dict.get(self.UNK_TOKEN, None)

        self.is_unk_in_vocab = self.UNK_INDEX is not None

        if self.UNK_INDEX is None:
            self.UNK_INDEX = total_predefined

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


class CustomVocab(Vocab):
    def __init__(self, vocab_path, embedding_file, data_dir=None):
        """Use this vocab class when you have a custom vocab as well as a
        custom embeddings file.

        This will inherit vocab class, so you will get predefined tokens with
        this one.

        IMPORTANT: To init your embedding, get your vectors from this class's
        object by calling `get_vectors` function

        Parameters
        ----------
        vocab_path : str
            Path of custom vocabulary
        embedding_file : str
            Path to custom embedding inititalization file
        data_dir : str
            Path to data directory if embedding file is not an absolute path.
            Default: None
        """
        super(CustomVocab, self).__init__(vocab_path)

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

        embedding_vectors = np.load(embedding_file)

        self.vectors = torch.FloatTensor(self.get_size(),
                                         len(embedding_vectors[0]))

        for i in range(0, 3):
            self.vectors[i] = torch.ones_like(self.vectors[i]) * 0.1 * i

        if not self.is_unk_in_vocab:
            self.vectors[self.UNK_INDEX] = \
                torch.ones_like(self.vectors[i]) * 0.1 * 3

        for i in range(4, self.get_size()):
            self.vectors[i] = embedding_vectors[i - 4]


class IntersectedVocab(Vocab):
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
        name = embedding_name.split('.')[1]
        dim = embedding_name.split('.')[2][:-1]

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

        embedding = getattr(vocab, class_name)(name, int(dim))

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


class PretrainedVocab(Vocab):
    def __init__(self, embedding_name):
        """Use this if you want to use pretrained embedding. See description
        of IntersectedVocab to get a list of the embedding available from
        torchtext

        Parameters
        ----------
        embedding_name : str
            Name of the pretrained alias for the embedding to used
        """
        name = embedding_name.split('.')[1]
        dim = embedding_name.split('.')[2][:-1]

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

        embedding = getattr(vocab, class_name)(name, int(dim))
        self.stoi = embedding.stoi
        self.itos = embedding.itos
        self.vectors = embedding.vectors
