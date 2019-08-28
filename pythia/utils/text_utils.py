# Copyright (c) Facebook, Inc. and its affiliates.
import os
import re
import torch
from itertools import chain
from collections import Counter

from pythia.utils.general import get_pythia_root, find_complete_inds, update_data

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


class TextDecoder:
    def __init__(self, vocab):
        self._vocab = vocab
        self._vocab_size = vocab.get_size()

        # Lists to store completed sequences and scores
        self._complete_seqs = []
        self._complete_seqs_scores = []

    def init_batch(self, sample_list):
        setattr(
            self,
            "seqs",
            sample_list.answers.new_full(
                (self._beam_size, 1), self._vocab.SOS_INDEX, dtype=torch.long
            ),
        )
        # Add a dim and duplicate the tensor beam_size times across that dim
        sample_list.image_feature_0 = (
            sample_list.image_feature_0.unsqueeze(1)
            .expand(-1, self._beam_size, -1, -1)
            .squeeze(0)
        )
        return sample_list

    def get_prob_from_scores(self, scores):
        return torch.nn.functional.log_softmax(scores, dim=1)

    def add_next_word(self, seqs, prev_word_inds, next_word_inds):
        return torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
        )

    def find_complete_inds(self, next_word_inds):
        incomplete_inds = []
        for ind, next_word in enumerate(next_word_inds):
            if next_word != self.vocab.EOS_INDEX:
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


class BeamSearch(TextDecoder):
    def __init__(self, vocab, beam_size=5):
        super().__init__(vocab)
        self._beam_size = beam_size

    def init_batch(self, sample_list):
        setattr(
            self,
            "top_k_scores",
            sample_list.answers.new_zeros((self._beam_size, 1), dtype=torch.float),
        )
        return super().init_batch(sample_list)

    def search(self, t, data, scores):
        # Add predicted scores to top_k_scores
        scores = self.get_prob_from_scores(scores)
        scores = self.top_k_scores.expand_as(scores) + scores

        # Find next top k scores and words. We flatten the scores tensor here
        # and get the top_k_scores and their indices top_k_words
        if t == 0:
            self.top_k_scores, top_k_words = scores[0].topk(
                self._beam_size, 0, True, True
            )
        else:
            self.top_k_scores, top_k_words = scores.view(-1).topk(
                self._beam_size, 0, True, True
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
        prev_word_inds = top_k_words // self._vocab_size
        next_word_inds = top_k_words % self._vocab_size

        # Add new words to sequences
        self.seqs = self.add_next_word(seqs, prev_word_inds, next_word_inds)

        # Find completed sequences
        complete_inds, incomplete_inds = self.find_complete_inds(next_word_inds)

        # Add to completed sequences
        if len(complete_inds) > 0:
            self._complete_seqs.extend(self.seqs[complete_inds].tolist())
            self._complete_seqs_scores.extend(self.top_k_scores[complete_inds])

        # Reduce beam length
        self._beam_size -= len(complete_inds)

        # Proceed with incomplete sequences
        if self._beam_size == 0:
            return True, data, 0

        self.seqs = self.seqs[incomplete_inds]
        self.top_k_scores = self.top_k_scores[incomplete_inds].unsqueeze(1)

        # TODO: Make the data update generic for any type of model
        # This is specific to BUTD model only.
        data = self.update_data(data, prev_word_inds, next_word_inds, incomplete_inds)

        next_beam_length = len(prev_word_inds[incomplete_inds])

        return False, data, next_beam_length

    def get_captions(self):
        if len(self._complete_seqs_scores) == 0:
            captions = torch.FloatTensor([0] * 5).unsqueeze(0)
        else:
            i = self._complete_seqs_scores.index(max(self._complete_seqs_scores))
            captions = torch.FloatTensor(self._complete_seqs[i]).unsqueeze(0)
        return captions

class NucleusSampling(TextDecoder):
    """Nucleus Sampling is a new text decoding strategy that avoids likelihood maximization.
    Rather, it works by sampling from the smallest set of top tokens which have a cumulative
    probability greater than a specified threshold.

    Present text decoding strategies like beam search do not work well on open-ended
    generation tasks (even on strong language models like GPT-2). They tend to repeat text
    a lot and the main reason behind it is that they try to maximize likelihood, which is a
    contrast from human-generated text which has a mix of high and low probability tokens.

    Nucleus Sampling is a stochastic approach and resolves this issue. Moreover, it improves
    upon other stochastic methods like top-k sampling by choosing the right amount of tokens
    to sample from. The overall result is better text generation on the same language model.
    """
    def __init__(self, vocab, threshold=0.8):
        super().__init__(vocab)
        # Threshold for sum of probability
        self._threshold = threshold

    def sample(self, t, data, scores):
        # Convert scores to probabilities
        scores = self.get_prob_from_scores(scores)
        # Sort scores in descending order and then select the top m elements having sum more than threshold.
        # We get the top_m_scores and their indices top_m_words
        if t == 0:
            top_m_scores, top_m_words = scores[0].sort(
                0, True
            )
        else:
            top_m_scores, top_m_words = scores.view(-1).sort(
                0, True
            )

        last_index = 0
        score_sum = 0
        for score in top_m_scores:
            last_index += 1
            score_sum += score
            if score_sum >= self._threshold:
                break

        top_m_scores = torch.div(top_m_scores[:last_index], score_sum)
        top_m_words = top_m_words[:last_index]
        # Zero value inside prev_word_inds because we are predicting a single stream of output.
        prev_word_ind = torch.tensor([0])
        # Get next word based on probabilities of top m words.
        next_word_ind = top_m_words[torch.multinomial(top_m_scores, 1)]
        # Add next word to sequence
        self.seqs = self.add_next_word(seqs, prev_word_ind, next_word_ind)
        # Check if sequence is complete
        complete_inds, incomplete_inds = self.find_complete_inds(next_word_ind)

        # If sequence is complete then return
        if len(complete_inds) > 0:
            self._complete_seq.extend(self.seq[complete_inds].tolist())
            return True, data, 0

        self.seq = self.seq[incomplete_inds]

        # TODO: Make the data update generic for any type of model
        # This is specific to BUTD model only.
        data = self.update_data(data, prev_word_ind, next_word_ind, incomplete_inds)

        return False, data, 1

    def get_caption(self):
        captions = torch.FloatTensor(self._complete_seq[0]).unsqueeze(0)
        return captions
