# Copyright (c) Facebook, Inc. and its affiliates.
import os
import re
import torch
from itertools import chain
from collections import Counter

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


def tokenize(sentence, regex=SENTENCE_SPLIT_REGEX, keep=["'s"], remove=[",", "?"]):
    sentence = sentence.lower()

    for token in keep:
        sentence = sentence.replace(token, " " + token)

    for token in remove:
        sentence = sentence.replace(token, "")

    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def word_tokenize(word, remove=[",", "?"]):
    word = word.lower()

    for item in remove:
        word = word.replace(item, "")
    word = word.replace("'s", " 's")

    return word.strip()


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


class VocabDict:
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

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
        self._build()

    def _build(self):
        if self.UNK_TOKEN not in self.word_list:
            self.word_list = [self.UNK_TOKEN] + self.word_list

        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}

        # String (word) to integer (index) dict mapping
        self.stoi = self.word2idx_dict
        # Integer to string (word) reverse mapping
        self.itos = self.word_list
        self.num_vocab = len(self.word_list)

        self.UNK_INDEX = (
            self.word2idx_dict[self.UNK_TOKEN] if self.UNK_TOKEN in self.word2idx_dict else None
        )

        self.PAD_INDEX = (
            self.word2idx_dict[self.PAD_TOKEN] if self.PAD_TOKEN in self.word2idx_dict else None
        )


    def idx2word(self, n_w):
        return self.word_list[n_w]

    def __len__(self):
        return len(self.word_list)

    def get_size(self):
        return len(self.word_list)

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return self.UNK_TOKEN

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


class VocabFromText(VocabDict):
    DEFAULT_TOKENS = [VocabDict.PAD_TOKEN, VocabDict.UNK_TOKEN,
                      VocabDict.START_TOKEN, VocabDict.END_TOKEN]

    def __init__(self, sentences, min_count=1, regex=SENTENCE_SPLIT_REGEX,
                 keep=[], remove=[], only_unk_extra=False):
        token_counter = Counter()

        for sentence in sentences:
            tokens = tokenize(
                sentence, regex=regex, keep=keep, remove=remove
            )
            token_counter.update(tokens)

        token_list = []
        for token in token_counter:
            if token_counter[token] >= min_count:
                token_list.append(token)

        extras = self.DEFAULT_TOKENS

        if only_unk_extra:
            extras = [self.UNK_TOKEN]

        self.word_list = extras + token_list
        self._build()


class BeamSearch:
    def __init__(self, vocab, beam_size=5):
        self.vocab = vocab
        self.vocab_size = vocab.get_size()
        self.beam_size = beam_size

        # Lists to store completed sequences and scores
        self.complete_seqs = []
        self.complete_seqs_scores = []

    def init_batch(self, sample_list):
        setattr(
            self,
            "seqs",
            sample_list.answers.new_full(
                (self.beam_size, 1), self.vocab.SOS_INDEX, dtype=torch.long
            ),
        )
        setattr(
            self,
            "top_k_scores",
            sample_list.answers.new_zeros((self.beam_size, 1), dtype=torch.float),
        )
        # Add a dim and duplicate the tensor beam_size times across that dim
        sample_list.image_feature_0 = (
            sample_list.image_feature_0.unsqueeze(1)
            .expand(-1, self.beam_size, -1, -1)
            .squeeze(0)
        )
        return sample_list

    def search(self, t, data, scores):
        # Add predicted scores to top_k_scores
        scores = torch.nn.functional.log_softmax(scores, dim=1)
        scores = self.top_k_scores.expand_as(scores) + scores

        # Find next top k scores and words. We flatten the scores tensor here
        # and get the top_k_scores and their indices top_k_words
        if t == 0:
            self.top_k_scores, top_k_words = scores[0].topk(
                self.beam_size, 0, True, True
            )
        else:
            self.top_k_scores, top_k_words = scores.view(-1).topk(
                self.beam_size, 0, True, True
            )

        # Convert to vocab indices. top_k_words contain indices from a flattened
        # k x vocab_size tensor. To get prev_word_indices we divide top_k_words
        # by vocab_size to determine which index in the beam among k generated
        # the next top_k_word. To get next_word_indices we take top_k_words
        # modulo vocab_size index. For example :
        # vocab_size : 9491
        # top_k_words : [610, 7, 19592, 9529, 292]
        # prev_word_inds : [0, 0, 2, 1, 0]
        # next_word_inds : [610, 7, 610, 38, 292]
        prev_word_inds = top_k_words // self.vocab_size
        next_word_inds = top_k_words % self.vocab_size

        # Add new words to sequences
        self.seqs = torch.cat(
            [self.seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
        )

        # Find completed sequences
        incomplete_inds = []
        for ind, next_word in enumerate(next_word_inds):
            if next_word != self.vocab.EOS_INDEX:
                incomplete_inds.append(ind)
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Add to completed sequences
        if len(complete_inds) > 0:
            self.complete_seqs.extend(self.seqs[complete_inds].tolist())
            self.complete_seqs_scores.extend(self.top_k_scores[complete_inds])

        # Reduce beam length
        self.beam_size -= len(complete_inds)

        # Proceed with incomplete sequences
        if self.beam_size == 0:
            return True, data, 0

        self.seqs = self.seqs[incomplete_inds]
        self.top_k_scores = self.top_k_scores[incomplete_inds].unsqueeze(1)

        # TODO: Make the data update generic for any type of model
        # This is specific to BUTD model only.
        data["texts"] = next_word_inds[incomplete_inds].unsqueeze(1)
        h1 = data["state"]["td_hidden"][0][prev_word_inds[incomplete_inds]]
        c1 = data["state"]["td_hidden"][1][prev_word_inds[incomplete_inds]]
        h2 = data["state"]["lm_hidden"][0][prev_word_inds[incomplete_inds]]
        c2 = data["state"]["lm_hidden"][1][prev_word_inds[incomplete_inds]]
        data["state"] = {"td_hidden": (h1, c1), "lm_hidden": (h2, c2)}

        next_beam_length = len(prev_word_inds[incomplete_inds])

        return False, data, next_beam_length

    def best_score(self):
        if len(self.complete_seqs_scores) == 0:
            captions = torch.FloatTensor([0] * 5).unsqueeze(0)
        else:
            i = self.complete_seqs_scores.index(max(self.complete_seqs_scores))
            captions = torch.FloatTensor(self.complete_seqs[i]).unsqueeze(0)
        return captions
