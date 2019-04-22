# Copyright (c) Facebook, Inc. and its affiliates.
import os
import re
from itertools import chain

from pythia.utils.general import get_pythia_root

SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")


def generate_ngrams(tokens, n=1):
    """Generate ngrams for particular 'n' from a list of tokens

    Parameters
    ----------
    tokens : List[str]
        List of tokens for which the ngram are to be generated
    n : int
        n for which ngrams are to be generated

    Returns
    -------
    List[str]
        List of ngrams generated

    """
    shifted_tokens = (tokens[i:] for i in range(n))
    tuple_ngrams = zip(*shifted_tokens)
    return (" ".join(i) for i in tuple_ngrams)


def generate_ngrams_range(tokens, ngram_range=(1, 3)):
    """Generates and returns a list of ngrams for all n present in ngram_range.

    Parameters
    ----------
    tokens : List[str]
        List of string tokens for which ngram are to be generated
    ngram_range : List[int]
        List of 'n' for which ngrams are to be generated. For e.g. if
        ngram_range = (1, 4) then it will returns 1grams, 2grams and 3grams

    Returns
    -------
    List[str]
        List of ngrams for each n in ngram_range.

    """
    assert len(ngram_range) == 2, (
        "'ngram_range' should be a tuple" " of two elements which is range of numbers"
    )
    return chain(*(generate_ngrams(tokens, i) for i in range(*ngram_range)))


def tokenize(sentence, regex=SENTENCE_SPLIT_REGEX):
    sentence = sentence.lower()
    sentence = sentence.replace(",", "").replace("?", "").replace("'s", " 's")
    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def word_tokenize(word):
    word = word.lower()
    word = word.replace(",", "").replace("?", "").replace("'s", " 's")
    return word.strip()


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


class VocabDict:
    def __init__(self, vocab_file, data_root_dir=None):
        if not os.path.isabs(vocab_file) and data_root_dir is not None:
            pythia_root = get_pythia_root()
            vocab_file = os.path.abspath(
                os.path.join(pythia_root, data_root_dir, vocab_file)
            )

        if not os.path.exists(vocab_file):
            raise RuntimeError(
                "Vocab file {} for vocab dict doesn't exist".format(vocab_file)
            )

        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.num_vocab = len(self.word_list)
        self.UNK_INDEX = (
            self.word2idx_dict["<unk>"] if "<unk>" in self.word2idx_dict else None
        )

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return "<unk>"

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_INDEX is not None:
            return self.UNK_INDEX
        else:
            raise ValueError(
                "word %s not in dictionary \
                             (while dictionary does not contain <unk>)"
                % w
            )

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds
