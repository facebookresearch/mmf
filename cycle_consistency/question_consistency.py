# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config.config import cfg


def build_question_consistency_module(**kwargs):
    return SentenceDecoder(**kwargs)


class SentenceDecoder(nn.Module):
    def __init__(self, **kwargs):
        super(SentenceDecoder, self).__init__()
        embed_size = kwargs.get('embed_size', 300)
        hidden_size = kwargs.get('hidden_size', 512)
        ans_embed_hidden_size = kwargs.get('ans_embed_hidden_size', 1000)
        image_feature_in_size = kwargs.get('image_feature_in_size', 2048)
# Add 2 for <start> and <end>
        q_vocab, a_vocab = self._get_vocabs()
        n_ans = len(a_vocab)
        vocab_size = len(q_vocab) + 2

        self.embed = nn.Embedding(vocab_size,
                                  embed_size,
                                  scale_grad_by_freq=False)

        embed_init_path = cfg.model.question_embedding[0]["par"]["embedding_init_file"]
        embed_init = np.load(embed_init_path)

        #initialize start and end at extremes
        se_init = np.zeros([2, embed_size])
        se_init[0][1] = 1.0
        se_init[1][1:] = -1.0
        embed_init = np.concatenate([embed_init, se_init], 0)
        self.embed.weight.data.copy_(torch.from_numpy(embed_init))

        self.img_embed = nn.Sequential(nn.Linear(image_feature_in_size, embed_size),
                                       nn.BatchNorm1d(embed_size, momentum=0.01))

        self.a_embed = nn.Sequential(nn.ReLU(),
                                     nn.Linear(n_ans, ans_embed_hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(ans_embed_hidden_size, embed_size),
                                     nn.ReLU())

        self.loss_fn = nn.CrossEntropyLoss()
        self.lstm = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = 14
        self.start_idx = vocab_size - 2
        self.end_idx = vocab_size - 1

    def _get_vocabs(self):
        q_vocab_path = os.path.join(cfg.data.data_root_dir,
                                    cfg.data.vocab_question_file)
        a_vocab_path = os.path.join(cfg.data.data_root_dir,
                                    cfg.data.vocab_answer_file)

        q_vocab = [l.rstrip() for l in tuple(open(q_vocab_path))]
        a_vocab = [l.rstrip() for l in tuple(open(a_vocab_path))]
        return q_vocab, a_vocab

    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt)

    def fuse_features(self, img_ft, ans_ft):
        img_ft = img_ft.to(self.img_embed[0].weight.device)
        ans_ft = ans_ft.to(self.a_embed[1].weight.device)
        img_feat = self.img_embed(img_ft)
        answer_embed = self.a_embed(ans_ft)

        mixed_feat = answer_embed + img_feat
        mixed_feat = mixed_feat.unsqueeze(1)
        return mixed_feat

    def forward(self, features, answer_logits, batch_tuple):
        mixed_feat = self.fuse_features(features, answer_logits)

        captions = batch_tuple[1]
        lengths = batch_tuple[0]['seq_length_batch'].clone().detach()
        captions = captions.to(self.embed.weight.device)

        # Add <end> token to captions
        for i in range(len(captions)):
            if lengths[i] <= self.max_seg_length -1:
                captions[i][lengths[i]] = self.end_idx
            else:
                captions[i][self.max_seg_length -1] = self.end_idx

        # Add <start> token to captions
        start_vector = torch.ones([len(captions), 1]).to(captions.device).long() * self.start_idx
        captions = torch.cat([start_vector, captions], 1)

        # Add 2 element to the lengths
        # Only needed if manually appending start and end vectors for original vocabulary
        lengths += 2 

        s_lengths, indices = torch.sort(lengths, descending=True)
        s_lengths = s_lengths.cpu().numpy().tolist()
        s_lengths = [s if s < self.max_seg_length else self.max_seg_length for s in s_lengths]
        captions = captions[indices]
        mixed_feat = mixed_feat[indices]

        embeddings = self.embed(captions)
        embeddings = torch.cat([mixed_feat, embeddings], 1)
        packed = pack_padded_sequence(embeddings, s_lengths, batch_first=True) 
        target_tuple = pack_padded_sequence(captions, s_lengths, batch_first=True)
        targets = target_tuple[0] 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])

        loss = self.compute_loss(outputs, targets)
        if not self.training:
            sampled_ids = self.sample(features, answer_logits)
        else:
            sampled_ids = self.sample(features, answer_logits)

        return {'q_token_pred': outputs,
                'qc_loss': loss,
                'q_token_gt': targets,
                'sampled_ids': sampled_ids,
                'answer_logits': answer_logits,
                'indices': indices}

    def sample(self, features, answer_logits, states=None):
        sampled_ids = []
        inputs = self.fuse_features(features, answer_logits)

        """ 
        To introduce noise in inference
        bs = features.shape[0]
        states = (torch.Tensor(1, bs, 512).normal_(0, 1.1).cuda(),
                  torch.Tensor(1, bs, 512).uniform_(0.5, -0.5).cuda())
        """

        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))    # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)               # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                 # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)        # sampled_ids: (batch_size, max_seq_length)

        return sampled_ids
