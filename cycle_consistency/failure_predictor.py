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

from config.config import cfg


def build_failure_prediction_module(input_size, **kwargs):
    return FailurePredictor(input_size, **kwargs)


class FailurePredictor(nn.Module):
    """Module that predicts the validity of a (I, Q, A) pair.
       
       I represents image features used by the VQA model
       Q represents the question representation of the VQA model
       A represents the distibution over the answer space predicted by the model
    """

    def __init__(self, input_size, **kwargs):
        super(FailurePredictor, self).__init__()

        self.combine_feats = kwargs["feat_combine"][:3] == "iqa"

        self.failure_predictor = nn.Sequential(
            nn.Linear(input_size, kwargs["hidden_1"]),
            nn.Dropout(p=kwargs["dropout"]),
            nn.ReLU(),
            nn.Linear(kwargs["hidden_1"], kwargs["hidden_2"]),
            nn.Dropout(p=kwargs["dropout"]),
            nn.ReLU(),
            nn.Linear(kwargs["hidden_2"], 2),
            nn.Softmax(dim=-1),
        )

        n_answers = self._get_n_options()

        s_embed_intermediate = kwargs["answer_hidden_size"]
        self.s_embed = nn.Sequential(
            nn.ReLU(), nn.Linear(n_answers, s_embed_intermediate), nn.ReLU()
        )

    def _get_n_options(self):
        a_vocab_path = os.path.join(cfg.data.data_root_dir, cfg.data.vocab_answer_file)
        a_vocab = [l.rstrip() for l in tuple(open(a_vocab_path))]
        return len(a_vocab)

    def forward(self, qi_embed, answer_emb):
        if self.combine_feats:
            answer_emb = self.s_embed(answer_emb["logits"])
            input_feat = torch.cat([qi_embed, answer_emb], -1)
        else:
            input_feat = qi_embed

        input_feat = input_feat.to(self.failure_predictor[0].weight.device)
        return {"confidence": self.failure_predictor(input_feat)}
