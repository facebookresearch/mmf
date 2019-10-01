# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from pythia.common.registry import registry
from pythia.modules.layers import ClassifierLayer
from pythia.models.pythia import Pythia
from pythia.utils.text_utils import BeamSearch, NucleusSampling


@registry.register_model("butd")
class BUTD(Pythia):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        self._build_word_embedding()
        self._init_feature_encoders("image")
        self._init_feature_embeddings("image")
        self._init_classifier()
        self._init_extras()

    def _build_word_embedding(self):
        self.text_processor = registry.get(self._datasets[0] + "_text_processor")
        self.vocab = self.text_processor.vocab
        self.vocab_size = self.vocab.get_size()
        self.word_embedding = self.vocab.get_embedding(
            torch.nn.Embedding, embedding_dim=self.config["embedding_dim"]
        )
        setattr(self, "text_embeddings_out_dim", self.config["embedding_dim"])

    def _init_classifier(self):
        self.classifier = ClassifierLayer(
            self.config["classifier"]["type"],
            in_dim=self.config["classifier"]["params"]["feature_dim"],
            out_dim=self.vocab_size,
            **self.config["classifier"]["params"]
        )

    def get_optimizer_parameters(self, config):
        params = [
            {"params": self.word_embedding.parameters()},
            {"params": self.image_feature_embeddings_list.parameters()},
            {"params": self.classifier.parameters()},
            {
                "params": self.image_feature_encoders.parameters(),
                "lr": (config["optimizer_attributes"]["params"]["lr"] * 0.1),
            },
        ]
        return params

    def prepare_data(self, sample_list, batch_size):
        setattr(self, "teacher_forcing", hasattr(sample_list, "text"))
        data = {}
        if self.teacher_forcing:
            caption_lengths, sort_ind = sample_list.caption_len.sort(
                dim=0, descending=True
            )
            data["decode_lengths"] = (caption_lengths - 1).tolist()
            sample_list.text = sample_list.text[sort_ind]
            sample_list.answers = sample_list.answers[sort_ind]
            sample_list.image_feature_0 = sample_list.image_feature_0[sort_ind]
            data["texts"] = sample_list.text
            timesteps = max(data["decode_lengths"])
            sample_list.add_field("targets", sample_list.text[:, 1:])
        else:
            data["texts"] = sample_list.answers.new_full(
                (batch_size, 1), self.vocab.SOS_INDEX, dtype=torch.long
            )
            timesteps = self.text_processor.max_length
            sample_list.add_field("targets", sample_list.answers[:, 0, 1:])
        return data, sample_list, timesteps

    def init_hidden_state(self, features):
        h = features.new_zeros(
            (features.size(0), self.config["classifier"]["params"]["hidden_dim"]),
            dtype=torch.float,
        )
        c = features.new_zeros(
            (features.size(0), self.config["classifier"]["params"]["hidden_dim"]),
            dtype=torch.float,
        )
        return h, c

    def get_data_t(self, t, data, batch_size_t, prev_output):
        if self.teacher_forcing:
            # Modify batch_size for timestep t
            batch_size_t = sum([l > t for l in data["decode_lengths"]])
        elif prev_output is not None and self.config["inference"]["type"] == "greedy":
            # Adding t-1 output words to data["text"] for greedy decoding
            output_softmax = torch.log_softmax(prev_output, dim=1)
            _, indices = torch.max(output_softmax, dim=1, keepdim=True)
            data["texts"] = torch.cat(
                (data["texts"], indices.view(batch_size_t, 1)), dim=1
            )

        # Slice data based on batch_size at timestep t
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
        registry.register("{}_lstm_state".format(h1.device), data["state"])

        return data, batch_size_t

    def forward(self, sample_list):
        # Stores the output probabilites.
        scores = sample_list.answers.new_ones(
            (
                sample_list.answers.size(0),
                self.text_processor.max_length,
                self.vocab_size,
            ),
            dtype=torch.float,
        )

        decoder = registry.get_decoder_class(self.config["inference"]["type"])(self.vocab, self.config)

        sample_list = decoder.init_batch(sample_list)
        # batch_size = sample_list.get_batch_size()
        batch_size = sample_list.image_feature_0.size(0)
        data, sample_list, timesteps = self.prepare_data(sample_list, batch_size)
        output = None
        batch_size_t = batch_size
        for t in range(timesteps):
            data, batch_size_t = self.get_data_t(t, data, batch_size_t, output)
            pi_t = data["texts"]
            embedding = self.word_embedding(pi_t)
            attention_feature, _ = self.process_feature_embedding(
                "image", sample_list, embedding[:, 0, :], batch_size_t=batch_size_t
            )
            output = self.classifier(attention_feature)
            # Compute decoding
            finish, data, batch_size_t = decoder.decode(t, data, output)
            if finish:
                break

        model_output = {"scores": scores}
        model_output["captions"] = decoder.get_result()

        return model_output
