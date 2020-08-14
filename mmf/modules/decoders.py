# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from mmf.common.registry import registry
from torch import nn
from torch.nn.utils.weight_norm import weight_norm


class VisDialDiscriminator(nn.Module):
    def __init__(self, config, embedding):
        super().__init__()
        self.config = config
        self.embedding = embedding

        self.emb_out_dim = embedding.text_out_dim
        self.hidden_dim = self.config.hidden_dim

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


class LanguageDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.language_lstm = nn.LSTMCell(
            in_dim + kwargs["hidden_dim"], kwargs["hidden_dim"], bias=True
        )
        self.fc = weight_norm(nn.Linear(kwargs["hidden_dim"], out_dim))
        self.dropout = nn.Dropout(p=kwargs["dropout"])
        self.init_weights(kwargs["fc_bias_init"])

    def init_weights(self, fc_bias_init):
        self.fc.bias.data.fill_(fc_bias_init)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, weighted_attn):
        # Get LSTM state
        state = registry.get(f"{weighted_attn.device}_lstm_state")
        h1, c1 = state["td_hidden"]
        h2, c2 = state["lm_hidden"]

        # Language LSTM
        h2, c2 = self.language_lstm(torch.cat([weighted_attn, h1], dim=1), (h2, c2))
        predictions = self.fc(self.dropout(h2))

        # Update hidden state for t+1
        state["lm_hidden"] = (h2, c2)

        return predictions
