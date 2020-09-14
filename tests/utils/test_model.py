# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from mmf.common.registry import registry
from torch import nn


class TestDecoderModel(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

    def build(self):
        return

    def init_hidden_state(self, features):

        h = features.new_zeros(
            (features.size(0), self.config.classifier.params.hidden_dim),
            dtype=torch.float,
        )
        c = features.new_zeros(
            (features.size(0), self.config.classifier.params.hidden_dim),
            dtype=torch.float,
        )
        return h, c

    def get_data_t(self, data, batch_size_t):
        data["texts"] = data["texts"][:batch_size_t]
        if "state" in data:
            h1 = data["state"]["td_hidden"][0][:batch_size_t]
            c1 = data["state"]["td_hidden"][1][:batch_size_t]
            h2 = data["state"]["lm_hidden"][0][:batch_size_t]
            c2 = data["state"]["lm_hidden"][1][:batch_size_t]
        else:
            h1, c1 = self.init_hidden_state(data["texts"])
            h2, c2 = self.init_hidden_state(data["texts"])
        data["state"] = {"td_hidden": (h1, c1), "lm_hidden": (h2, c2)}
        registry.register(f"{h1.device}_lstm_state", data["state"])

        return data, batch_size_t

    def forward(self, sample_list):
        scores = torch.rand(sample_list.get_batch_size(), 3127)
        decoder = registry.get_decoder_class(self.config.inference.type)(
            self.vocab, self.config
        )
        sample_list.add_field("targets", sample_list.answers[:, 0, 1:])
        sample_list = decoder.init_batch(sample_list)
        batch_size = sample_list.image_feature_0.size(0)
        data = {}
        data["texts"] = sample_list.answers.new_full(
            (batch_size, 1), self.vocab.SOS_INDEX, dtype=torch.long
        )
        timesteps = 10
        output = None
        batch_size_t = batch_size
        for t in range(timesteps):
            data, batch_size_t = self.get_data_t(data, batch_size_t)
            output = torch.randn(batch_size_t, self.vocab.get_size())
            if t == timesteps - 1:
                # manually add EOS to the first example.
                output = torch.ones(batch_size_t, self.vocab.get_size()) * -30.0
                output[0, self.vocab.EOS_INDEX] = 10
            finish, data, batch_size_t = decoder.decode(t, data, output)
            if finish:
                break

        model_output = {"scores": scores}
        model_output["captions"] = decoder.get_result()

        return model_output
