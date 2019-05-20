# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from pythia.common.registry import registry
from pythia.modules.encoders import ImageEncoder
from pythia.modules.decoders import CaptioningDecoder
from torch import nn

from .base_model import BaseModel


@registry.register_model("butd")
class BUTD(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    def build(self):
        self._build_word_embedding()
        self._init_feature_encoders("image")
        self._init_decoder()
        self._init_extras()

    def _build_word_embedding(self):
        text_processor = registry.get(self._datasets[0] + "_text_processor")
        self.vocab = text_processor.vocab
        self.vocab_size = self.vocab.get_size()
        self.word_embedding = self.vocab.get_embedding(
            torch.nn.Embedding, embedding_dim=300
        )

    def _init_feature_encoders(self, attr):
        feat_encoders = []
        feat_encoders_list_config = self.config[
            attr + "_feature_encodings"
        ]  # finetune_faster_rcnn_fpn_fc7
        feature_dim = self.config[attr + "_feature_dim"]  # 2048
        setattr(self, attr + "_feature_dim", feature_dim)

        for feat_encoder in feat_encoders_list_config:
            encoder_type = feat_encoder["type"]
            encoder_kwargs = feat_encoder["params"]
            encoder_kwargs["model_data_dir"] = self.config["model_data_dir"]

            feat_model = ImageEncoder(encoder_type, feature_dim, **encoder_kwargs)

            feat_encoders.append(feat_model)
            setattr(self, attr + "_feature_dim", feat_model.out_dim)

        setattr(self, attr + "_feature_encoders", nn.ModuleList(feat_encoders))

    def _init_decoder(self):
        self.decoder = CaptioningDecoder(
            attention_dim=1024,
            embed_dim=300,
            decoder_dim=1024,
            vocab_size=self.vocab_size,
        )

    def _init_extras(self):
        self.inter_model = None

    def get_optimizer_parameters(self, config):
        params = [
            {"params": self.word_embedding.parameters()},
            {"params": self.decoder.parameters()},
            {
                "params": self.image_feature_encoders.parameters(),
                "lr": (config["optimizer_attributes"]["params"]["lr"] * 0.1),
            },
        ]

        return params

    def process_feature_embedding(self, attr, sample_list, extra=[]):
        feature_embeddings = []
        features = []

        # Convert list of keys to the actual values
        extra = sample_list.get_fields(extra)

        feature_idx = 0

        # Get all of the features, which are in the form, "image_feature_0"
        # "image_feature_1" ...
        while True:
            feature = getattr(
                sample_list, "{}_feature_{:d}".format(attr, feature_idx), None
            )
            if feature is None:
                break
            feature_idx += 1
            features.append(feature)

        feature_encoders = getattr(self, attr + "_feature_encoders")
        # Each feature should have a separate image feature encoders
        assert len(features) == len(feature_encoders), (
            "Number of feature encoders, {} are not equal "
            "to number of features, {}.".format(len(feature_encoders), len(features))
        )

        # Now, iterate to get final image features
        for i, feature in enumerate(features):

            # Attribute in which encoders are saved, for "image" it
            # will be "image_feature_encoders", other example is
            # "context_feature_encoders"
            encoders_attr = attr + "_feature_encoders"
            feature_encoder = getattr(self, encoders_attr)[i]

            # Encode the features
            encoded_feature = feature_encoder(feature)
            feature_embeddings.append(encoded_feature)

        # Concatenate all features embeddings
        feature_embedding_total = torch.cat(feature_embeddings, dim=1)
        return feature_embedding_total

    def get_data_t(self, data, batch_size_t):
        data["texts"] = data["texts"][:batch_size_t]
        data["image_features"] = data["image_features"][:batch_size_t]

        if "state" in data:
            h1 = data["state"]["td_hidden"][0][:batch_size_t]
            c1 = data["state"]["td_hidden"][1][:batch_size_t]
            h2 = data["state"]["lm_hidden"][0][:batch_size_t]
            c2 = data["state"]["lm_hidden"][1][:batch_size_t]
            data["state"] = {"td_hidden": (h1, c1), "lm_hidden": (h2, c2)}
        return data

    def forward(self, sample_list):
        image_embedding_total = self.process_feature_embedding("image", sample_list)

        teacher_forcing = hasattr(sample_list, "text")
        data = {}
        if teacher_forcing:
            caption_lengths, sort_ind = sample_list.text_len.sort(
                dim=0, descending=True
            )
            decode_lengths = (caption_lengths - 1).tolist()
            sample_list.text = sample_list.text[sort_ind]
            sample_list.answers = sample_list.answers[sort_ind]
            data["texts"] = sample_list.text
            data["image_features"] = image_embedding_total[sort_ind]
            timesteps = max(decode_lengths)
        else:
            data["texts"] = torch.LongTensor(
                [[self.vocab.SOS_INDEX]] * sample_list.get_batch_size()
            ).to("cuda")
            data["image_features"] = image_embedding_total
            timesteps = 52

        batch_size = sample_list.get_batch_size()

        scores = torch.ones(batch_size, timesteps, self.vocab_size).to("cuda")

        for t in range(timesteps):
            if not teacher_forcing:
                batch_size_t = batch_size
            else:
                batch_size_t = sum([l > t for l in decode_lengths])
            data = self.get_data_t(data, batch_size_t)
            pi_t = data["texts"][:, t].unsqueeze(-1)
            embedding = self.word_embedding(pi_t)
            if "state" not in data:
                data["state"] = None
            output, data["state"] = self.decoder(
                data["image_features"], embedding[:, 0, :], data["state"]
            )
            scores[:batch_size_t, t] = output

            # Greedy decoding
            if not teacher_forcing:
                output_softmax = torch.log_softmax(output, dim=1)
                _, indices = torch.max(output_softmax, dim=1, keepdim=True)
                data["texts"] = torch.cat(
                    (data["texts"], indices.view(batch_size_t, 1)), dim=1
                )

        targets = (
            sample_list.answers[:, 0, 1:]
            if not teacher_forcing
            else sample_list.text[:, 1:]
        )
        sample_list.add_field("targets", targets)
        sample_list.add_field("scores", scores)
        model_output = {"scores": scores, "targets": targets}

        return model_output