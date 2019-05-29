# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn


class VisDialDiscriminator(nn.Module):
    def __init__(self, config, embedding):
        super(VisDialDiscriminator, self).__init__()
        self.config = config
        self.embedding = embedding

        self.emb_out_dim = embedding.text_out_dim
        self.hidden_dim = self.config["hidden_dim"]

        self.projection_layer = nn.Linear(self.emb_out_dim, self.hidden_dim)

    def forward(self, encoder_output, batch):
        answer_options_len = batch["answer_options_len"]

        # BATCH_SIZE X DIALOGUES X 100 X SEQ_LEN
        answer_options = batch["answer_options"]

        max_seq_len = answer_options.size(-1)

        batch_size, ndialogues, noptions, seq_len = answer_options.size()

        # (B X D X 100) X SEQ_LEN
        answer_options = answer_options.view(-1, max_seq_len)
        answer_options_len = answer_options_len.view(-1)

        # (B x D x 100) x EMB_OUT_DIM
        answer_options = self.embedding(answer_options)

        # (B x D x 100) x HIDDEN_DIM
        answer_options = self.projection_layer(answer_options)

        # (B x D) x 100 x HIDDEN_DIM
        answer_options = answer_options.view(
            batch_size * ndialogues, noptions, self.hidden_dim
        )

        # (B x D) x HIDDEN_DIM => (B x D) x 100 x HIDDEN_DIM
        encoder_output = encoder_output.unsqueeze(1).expand(-1, noptions, -1)

        # (B x D) x 100 x HIDDEN_DIM * (B x D) x 100 x HIDDEN_DIM = SAME THING
        # SUM => (B x D) x 100
        scores = torch.sum(answer_options * encoder_output, dim=2)

        return scores
