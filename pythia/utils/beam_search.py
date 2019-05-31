# Copyright (c) Facebook, Inc. and its affiliates.

import torch


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
