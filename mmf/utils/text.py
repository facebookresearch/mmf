# Copyright (c) Facebook, Inc. and its affiliates.
"""
Text utils module contains implementations for various decoding strategies like
Greedy, Beam Search and Nucleus Sampling.

In your model's config you can specify ``inference`` attribute to use these strategies
in the following way:

.. code::

   model_config:
       some_model:
           inference:
               - type: greedy
               - params: {}
"""
import os
import re
from collections import Counter
from itertools import chain

import torch
from mmf.common.registry import registry
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_absolute_path


SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")


def generate_ngrams(tokens, n=1):
    """Generate ngrams for particular 'n' from a list of tokens

    Args:
        tokens (List[str]): List of tokens for which the ngram are to be generated
        n (int, optional): n for which ngrams are to be generated. Defaults to 1.

    Returns:
        List[str]: List of ngrams generated.
    """
    shifted_tokens = (tokens[i:] for i in range(n))
    tuple_ngrams = zip(*shifted_tokens)
    return (" ".join(i) for i in tuple_ngrams)


def generate_ngrams_range(tokens, ngram_range=(1, 3)):
    """Generates and returns a list of ngrams for all n present in ngram_range

    Args:
        tokens (List[str]): List of string tokens for which ngram are to be generated
        ngram_range (List[int], optional): List of 'n' for which ngrams are to be
            generated. For e.g. if ngram_range = (1, 4) then it will returns
            1grams, 2grams and 3grams. Defaults to (1, 3).

    Returns:
        List[str]: List of ngrams for each n in ngram_range
    """
    assert len(ngram_range) == 2, (
        "'ngram_range' should be a tuple" " of two elements which is range of numbers"
    )
    return chain(*(generate_ngrams(tokens, i) for i in range(*ngram_range)))


def tokenize(sentence, regex=SENTENCE_SPLIT_REGEX, keep=None, remove=None):
    if keep is None:
        keep = ["'s"]
    if remove is None:
        remove = [",", "?"]
    sentence = sentence.lower()

    for token in keep:
        sentence = sentence.replace(token, " " + token)

    for token in remove:
        sentence = sentence.replace(token, "")

    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def word_tokenize(word, remove=None):
    if remove is None:
        remove = [",", "?"]
    word = word.lower()

    for item in remove:
        word = word.replace(item, "")
    word = word.replace("'s", " 's")

    return word.strip()


def load_str_list(fname):
    with PathManager.open(fname) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


class VocabDict:
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, vocab_file, data_dir=None):
        if not os.path.isabs(vocab_file) and data_dir is not None:
            vocab_file = get_absolute_path(os.path.join(data_dir, vocab_file))

        if not PathManager.exists(vocab_file):
            raise RuntimeError(f"Vocab file {vocab_file} for vocab dict doesn't exist")

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
            self.word2idx_dict[self.UNK_TOKEN]
            if self.UNK_TOKEN in self.word2idx_dict
            else None
        )

        self.PAD_INDEX = (
            self.word2idx_dict[self.PAD_TOKEN]
            if self.PAD_TOKEN in self.word2idx_dict
            else None
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
    DEFAULT_TOKENS = [
        VocabDict.PAD_TOKEN,
        VocabDict.UNK_TOKEN,
        VocabDict.START_TOKEN,
        VocabDict.END_TOKEN,
    ]

    def __init__(
        self,
        sentences,
        min_count=1,
        regex=SENTENCE_SPLIT_REGEX,
        keep=None,
        remove=None,
        only_unk_extra=False,
    ):
        if keep is None:
            keep = []
        if remove is None:
            remove = []
        token_counter = Counter()

        for sentence in sentences:
            tokens = tokenize(sentence, regex=regex, keep=keep, remove=remove)
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


class TextDecoder:
    """Base class to be inherited by all decoding strategies. Contains
    implementations that are common for all strategies.

    Args:
        vocab (list): Collection of all words in vocabulary.

    """

    def __init__(self, vocab):
        self._vocab = vocab
        self._vocab_size = vocab.get_size()

        # Lists to store completed sequences and scores
        self._complete_seqs = []
        self._complete_seqs_scores = []

    def init_batch(self, sample_list):
        img_size = sample_list.image_feature_0.size()
        self._batch_size, feature_size_1, feature_size_2 = img_size
        t_batch_size = self._batch_size * self._decode_size
        self.seqs = sample_list.answers.new_full(
            (t_batch_size, 1), self._vocab.SOS_INDEX, dtype=torch.long
        )
        sample_list.image_feature_0 = (
            sample_list.image_feature_0.unsqueeze(1)
            .expand(-1, self._decode_size, -1, -1)
            .reshape(t_batch_size, feature_size_1, feature_size_2)
        )
        self.sample_list = sample_list
        return sample_list

    def add_next_word(self, seqs, prev_word_inds, next_word_inds):
        return torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

    def find_complete_inds(self, next_word_inds):
        incomplete_inds = []
        for ind, next_word in enumerate(next_word_inds):
            if next_word != self._vocab.EOS_INDEX:
                incomplete_inds.append(ind)
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        return complete_inds, incomplete_inds

    def update_data(self, data, prev_word_inds, next_word_inds, incomplete_inds):
        data["texts"] = next_word_inds[incomplete_inds].unsqueeze(1)
        h1 = data["state"]["td_hidden"][0][prev_word_inds[incomplete_inds]]
        c1 = data["state"]["td_hidden"][1][prev_word_inds[incomplete_inds]]
        h2 = data["state"]["lm_hidden"][0][prev_word_inds[incomplete_inds]]
        c2 = data["state"]["lm_hidden"][1][prev_word_inds[incomplete_inds]]
        data["state"] = {"td_hidden": (h1, c1), "lm_hidden": (h2, c2)}
        return data


@registry.register_decoder("beam_search")
class BeamSearch(TextDecoder):
    def __init__(self, vocab, config):
        super().__init__(vocab)
        self._decode_size = config["inference"]["params"]["beam_length"]

    def init_batch(self, sample_list):
        self.sample_list = super().init_batch(sample_list)

        # initialize with t_batch_size = _batch_size * _decode_size
        self.top_k_scores = sample_list.answers.new_zeros(
            (self._batch_size * self._decode_size, 1), dtype=torch.float
        )
        # maintain _decode_size, _complete_seqs and _complete_seqs_scores
        # for each example in a batch.
        self._decode_sizes = [self._decode_size] * self._batch_size
        for _ in range(self._batch_size):
            self._complete_seqs.append([])
            self._complete_seqs_scores.append([])
        return self.sample_list

    def decode(self, t, data, scores):
        # Add predicted scores to top_k_scores
        scores = torch.nn.functional.log_softmax(scores, dim=1)
        scores = self.top_k_scores.expand_as(scores) + scores

        # Find next top k scores and words. We flatten the scores tensor here
        # and get the top_k_scores and their indices top_k_words
        top_k_scores, top_k_words = [], []
        ex_start = 0
        for decode_size in self._decode_sizes:
            ex_end = ex_start + decode_size
            if t == 0:
                top_k_score, top_k_word = scores[ex_start].topk(
                    decode_size, 0, True, True
                )
            else:
                top_k_score, top_k_word = (
                    scores[ex_start:ex_end].view(-1).topk(decode_size, 0, True, True)
                )
            top_k_scores.extend(top_k_score)
            top_k_words.append(top_k_word)
            ex_start = ex_end
        self.top_k_scores = torch.stack(top_k_scores)
        # Convert to vocab indices. top_k_words contain indices from a flattened
        # k x vocab_size tensor. To get prev_word_indices we divide top_k_words
        # by vocab_size to determine which index in the beam among k generated
        # the next top_k_word. To get next_word_indices we take top_k_words
        # modulo vocab_size index. For example :
        # vocab_size : 9491
        # top_k_words : [610, 7, 19592, 9529, 292]
        # prev_word_ind : [0, 0, 2, 1, 0]
        # next_word_ind : [610, 7, 610, 38, 292]
        # further, shift the prev_word_ind by ex_start to find corresponding example
        # within a batch.

        ex_start = 0
        prev_word_inds, next_word_inds = [], []
        for ex_idx, decode_size in enumerate(self._decode_sizes):
            prev_word_inds.extend((top_k_words[ex_idx] // self._vocab_size) + ex_start)
            next_word_inds.extend(top_k_words[ex_idx] % self._vocab_size)
            ex_start += decode_size
        prev_word_inds = torch.stack(prev_word_inds)
        next_word_inds = torch.stack(next_word_inds)

        # Add new words to sequences
        self.seqs = self.add_next_word(self.seqs, prev_word_inds, next_word_inds)
        # Find completed sequences
        complete_inds, incomplete_inds = self.find_complete_inds(next_word_inds)

        # Add to completed sequences and Reduce beam length
        ex_start = 0
        for ex_idx, decode_size in enumerate(self._decode_sizes):
            for beam_idx in range(ex_start, ex_start + decode_size):
                if beam_idx in complete_inds:
                    top_k_score = self.top_k_scores[beam_idx]
                    self._complete_seqs[ex_idx].append(self.seqs[beam_idx].tolist())
                    self._complete_seqs_scores[ex_idx].append(top_k_score)
                    self._decode_sizes[ex_idx] -= 1
            ex_start += decode_size

        # Proceed with incomplete sequences
        if sum(self._decode_sizes) == 0:
            return True, data, 0
        self.seqs = self.seqs[incomplete_inds]
        self.top_k_scores = self.top_k_scores[incomplete_inds].unsqueeze(1)

        # TODO: Make the data update generic for any type of model
        # This is specific to BUTD model only.
        image_feature_0 = self.sample_list.image_feature_0
        self.sample_list.image_feature_0 = image_feature_0[incomplete_inds]
        data = self.update_data(data, prev_word_inds, next_word_inds, incomplete_inds)

        next_beam_length = len(prev_word_inds[incomplete_inds])

        return False, data, next_beam_length

    def get_result(self):
        captions = []
        max_len = 0
        for ex_idx in range(len(self._complete_seqs_scores)):
            if len(self._complete_seqs_scores[ex_idx]) == 0:
                captions.append([0] * 5)
                max_len = max(5, max_len)
            else:
                max_score = max(self._complete_seqs_scores[ex_idx])
                max_idx = self._complete_seqs_scores[ex_idx].index(max_score)
                captions.append(self._complete_seqs[ex_idx][max_idx])
                max_len = max(max_len, len(captions[-1]))
        for ex_idx in range(len(captions)):
            padded_tokens = [self._vocab.PAD_INDEX] * (max_len - len(captions[ex_idx]))
            captions[ex_idx].extend(padded_tokens)
        return torch.FloatTensor(captions)


@registry.register_decoder("nucleus_sampling")
class NucleusSampling(TextDecoder):
    """Nucleus Sampling is a new text decoding strategy that avoids likelihood
    maximization. Rather, it works by sampling from the smallest set of top
    tokens which have a cumulative probability greater than a specified
    threshold.

    Present text decoding strategies like beam search do not work well on open-ended
    generation tasks (even on strong language models like GPT-2). They tend to repeat
    text a lot and the main reason behind it is that they try to maximize likelihood,
    which is a contrast from human-generated text which has a mix of high and low
    probability tokens.

    Nucleus Sampling is a stochastic approach and resolves this issue. Moreover,
    it improves upon other stochastic methods like top-k sampling by choosing the
    right amount of tokens to sample from. The overall result is better text
    generation on the same language model.

    Link to the paper introducing Nucleus Sampling (Section 6) -
    https://arxiv.org/pdf/1904.09751.pdf

    Args:
        vocab (list): Collection of all words in vocabulary.
        sum_threshold (float): Ceiling of sum of probabilities of tokens to
            sample from.
    """

    def __init__(self, vocab, config):
        super().__init__(vocab)
        self._decode_size = 1
        # Threshold for sum of probability
        self._threshold = config["inference"]["params"]["sum_threshold"]

    def decode(self, t, data, scores):
        # Convert scores to probabilities
        scores = torch.nn.functional.softmax(scores, dim=1)
        # Sort scores in descending order and then select the top m elements having
        # sum more than threshold.
        # We get the top_m_scores and their indices top_m_words
        if t == 0:
            top_m_scores, top_m_words = scores[0].sort(0, True)
        else:
            top_m_scores, top_m_words = scores.view(-1).sort(0, True)

        last_index = 0
        score_sum = 0
        for score in top_m_scores:
            last_index += 1
            score_sum += score
            if score_sum >= self._threshold:
                break

        top_m_scores = torch.div(top_m_scores[:last_index], score_sum)
        top_m_words = top_m_words[:last_index]

        # Zero value inside prev_word_inds because we are predicting a single
        # stream of output.
        prev_word_ind = torch.tensor([0])
        # Get next word based on probabilities of top m words.
        next_word_ind = top_m_words[torch.multinomial(top_m_scores, 1)]
        # Add next word to sequence

        self.seqs = self.add_next_word(self.seqs, prev_word_ind, next_word_ind)
        # Check if sequence is complete
        complete_inds, incomplete_inds = self.find_complete_inds(next_word_ind)
        # If sequence is complete then return
        if len(complete_inds) > 0:
            self._complete_seqs.extend(self.seqs[complete_inds].tolist())
            return True, data, 0

        self.seqs = self.seqs[incomplete_inds]

        data = self.update_data(data, prev_word_ind, next_word_ind, incomplete_inds)

        return False, data, 1

    def get_result(self):
        if len(self._complete_seqs) == 0:
            captions = torch.FloatTensor([0] * 5).unsqueeze(0)
        else:
            captions = torch.FloatTensor(self._complete_seqs[0]).unsqueeze(0)
        return captions
