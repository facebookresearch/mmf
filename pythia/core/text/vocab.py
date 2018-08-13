import os
import sys
import torch

from collections import defaultdict
from torchtext import vocab


class Vocab:
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'
    UNK_TOKEN = '<unk>'

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, vocab_path):
        if not os.path.exists(vocab_path):
            print("Vocab not found at " + vocab_path)
            sys.exit(1)

        self.itos = [''] * 4
        self.itos[self.PAD_INDEX] = self.PAD_TOKEN
        self.itos[self.UNK_INDEX] = self.UNK_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN

        # Return unk index by default
        self.stoi = defaultdict(lambda: self.UNK_INDEX)
        self.stoi[self.SOS_TOKEN] = self.SOS_INDEX
        self.stoi[self.EOS_TOKEN] = self.EOS_INDEX
        self.stoi[self.UNK_TOKEN] = self.UNK_INDEX
        self.stoi[self.PAD_INDEX] = self.PAD_INDEX

        index = len(self.itos)

        with open(vocab_path, 'r') as f:
            for line in f:
                self.itos.append(line.strip())
                self.stoi[line.strip()] = index
                index += 1

    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi

    def get_size(self):
        return len(self.itos)


class GloVeIntersectedVocab(Vocab):
    def __init__(self, vocab_file, embedding_name):
        super(GloVeIntersectedVocab, self).__init__(vocab_file)
        name = embedding_name.split('.')[1]
        dim = embedding_name.split('.')[2][:-1]
        glove = vocab.GloVe(name, int(dim))

        self.vectors = torch.FloatTensor(self.get_size(),
                                         len(glove.vectors[0]))
        self.vectors[0].zero_()

        for i in range(1, 4):
            self.vectors[i] = torch.ones_like(self.vectors[i]) * 0.1 * i

        for i in range(4, self.get_size()):
            word = self.itos[i]
            glove_index = glove.stoi.get(word, None)

            if glove_index is None:
                self.vectors[i] = self.vectors[self.UNK_INDEX].clone()
            else:
                self.vectors[i] = glove.vectors[glove_index]
