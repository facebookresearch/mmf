# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn

from .attention import CaptionAttention


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


class CaptioningDecoder(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        features_dim=2048,
        dropout=0.5,
    ):
        super(CaptioningDecoder, self).__init__()

        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = CaptionAttention(
            features_dim, decoder_dim, attention_dim
        )

        self.dropout = nn.Dropout(p=self.dropout)
        self.top_down_attention = nn.LSTMCell(
            embed_dim + features_dim + decoder_dim, decoder_dim, bias=True
        )
        self.language_model = nn.LSTMCell(
            features_dim + decoder_dim, decoder_dim, bias=True
        )
        self.fc = nn.utils.weight_norm(
            nn.Linear(decoder_dim, vocab_size)
        )
        self.init_weights()
     
    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, batch_size):
        h = torch.zeros(batch_size, self.decoder_dim).to("cuda")
        c = torch.zeros(batch_size, self.decoder_dim).to("cuda")
        return h, c

    def forward(self, image_features, caption_embeddings, state=None):
        # Mean of image features
        image_features_mean = image_features.mean(1)

        # Initialize LSTM state
        if state:
            h1, c1 = state["td_hidden"]
            h2, c2 = state["lm_hidden"]
        else:
            h1, c1 = self.init_hidden_state(image_features.size(0))
            h2, c2 = self.init_hidden_state(image_features.size(0))

        # Top Down Attention LSTM
        h1, c1 = self.top_down_attention(
            torch.cat([h2, image_features_mean, caption_embeddings], dim=1), (h1, c1)
        )
        # Attention layer
        attention_weighted_encoding = self.attention(image_features, h1)

        # Language LSTM
        h2, c2 = self.language_model(
            torch.cat([attention_weighted_encoding, h1], dim=1), (h2, c2)
        )
        predictions = self.fc(self.dropout(h2))

        # Update hidden state for t+1
        state = {"td_hidden": (h1, c1), "lm_hidden": (h2, c2)}

        return predictions, state
