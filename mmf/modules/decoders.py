# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

from mmf.common.registry import registry
from mmf.modules.embeddings import TextEmbedding


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


class PyTorchTransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        attention_heads: int,
        feedforward_size: int,
        dropout: float = 0.1,
        padding_idx: int = 0,
        max_sequence_length: int = 30,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.padding_idx = padding_idx

        # Create a randomly initialized word + positional embedding.
        self.embedding = TextEmbedding(
            "wordandpos",
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            max_sequence_length=max_sequence_length,
            dropout=dropout,
            padding_idx=padding_idx,
        )
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                self.hidden_size,
                self.attention_heads,
                dim_feedforward=self.feedforward_size,
                dropout=dropout,
                activation="gelu",
            ),
            self.num_layers
        )
        self.apply(self.init_weights)

        # Create an output linear layer and tie the input and output word
        # embeddings to reduce parameters.
        self.output = nn.Linear(self.textual_feature_size, vocab_size)
        self.output.weight = self.embedding.words.weight

    @staticmethod
    def init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        encoded_features: torch.Tensor,
        sequence_tokens: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_sequence_length = sequence_tokens.size()

        # Create a mask based on sequence lengths, shape: (batch_size, )
        # Form a binary mask: it is True for padding positions.
        # These positions will be ignored for multi-headed attention.
        ones = torch.ones_like(sequence_tokens)
        sequence_mask = sequence_lengths.unsqueeze(1) < ones.cumsum(dim=1)

        # shape: (batch_size, max_sequence_length, textual_feature_size)
        embeddings = self.embedding(sequence_tokens)

        # An additive mask for masking the future (one direction).
        unidirectional_mask = self._generate_future_mask(
            max_sequence_length, embeddings.dtype, embeddings.device
        )
        # We transpose the first two dimensions of tokens embeddings and encoded
        # features, as required by transformer.
        embeddings = embeddings.transpose(0, 1)
        encoded_features = encoded_features.transpose(0, 1)

        # shape: (max_sequence_length, batch_size, hidden_size)
        textual_features = self.transformer(
            embeddings,
            encoded_features,
            tgt_mask=unidirectional_mask,
            tgt_key_padding_mask=sequence_mask,
        )
        # Undo the transpose and bring batch to dim 0.
        # shape: (batch_size, max_sequence_length, hidden_size)
        textual_features = textual_features.transpose(0, 1)

        # shape: (batch_size, max_sequence_length, vocab_size)
        output_logits = self.output(textual_features)
        return output_logits

    def _generate_future_mask(
        self, size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        r"""
        Generate a mask for "future" positions, useful when using this module
        for language modeling.

        Parameters
        ----------
        size: int
        """
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=dtype), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask
