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
        self.text_processor = registry.get(self._datasets[0] + "_text_processor")
        self.vocab = self.text_processor.vocab
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
            caption_lengths, sort_ind = sample_list.caption_len.sort(
                dim=0, descending=True
            )
            decode_lengths = (caption_lengths - 1).tolist()
            sample_list.text = sample_list.text[sort_ind]
            sample_list.answers = sample_list.answers[sort_ind]
            data["texts"] = sample_list.text
            data["image_features"] = image_embedding_total[sort_ind]
            timesteps = max(decode_lengths)
        else:
            data["texts"] = image_embedding_total.new_full(
                (sample_list.get_batch_size(), 1),
                self.vocab.SOS_INDEX,
                dtype=torch.long,
            )
            data["image_features"] = image_embedding_total
            timesteps = self.text_processor.max_length

        batch_size = image_embedding_total.size(0)

        scores = image_embedding_total.new_ones(
            (batch_size, timesteps, self.vocab_size)
        )

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
        model_output = {"scores": scores}

        return model_output

    def get_data_beam_search_t(self, data, batch_size_t):
        data["image_features"] = data["image_features"][batch_size_t]
        if "state" in data:
            h1 = data["state"]["td_hidden"][0][batch_size_t]
            c1 = data["state"]["td_hidden"][1][batch_size_t]
            h2 = data["state"]["lm_hidden"][0][batch_size_t]
            c2 = data["state"]["lm_hidden"][1][batch_size_t]
            data["state"] = {"td_hidden": (h1, c1), "lm_hidden": (h2, c2)}
        return data

    def beam_search(self, sample_list, k=5):
        image_embedding_total = self.process_feature_embedding("image", sample_list)

        top_k_scores = image_embedding_total.new_zeros((k, 1))
        data = {}
        data["texts"] = image_embedding_total.new_full(
            (k, 1), self.vocab.SOS_INDEX, dtype=torch.long
        )
        data["image_features"] = (
            image_embedding_total.unsqueeze(1).expand(-1, k, -1, -1).squeeze(0)
        )
        seqs = data["texts"]
        timesteps = self.text_processor.max_length

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Dummy output for loss calculation to not complain
        outputs = image_embedding_total.new_ones(
            (sample_list.get_batch_size(), timesteps, self.vocab_size)
        )

        for t in range(timesteps):
            data["texts"] = self.word_embedding(data["texts"])
            data["state"] = data.pop("state", None)

            scores, data["state"] = self.decoder(
                data["image_features"], data["texts"][:, 0, :], data["state"]
            )

            # Add predicted scores to top_k_scores
            scores = torch.nn.functional.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            # Find next top k scores and words
            if t == 0:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            # Convert to vocab indices
            prev_word_inds = top_k_words / self.vocab_size
            next_word_inds = top_k_words % self.vocab_size

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            # Find completed sequences
            incomplete_inds = [
                ind
                for ind, next_word in enumerate(next_word_inds)
                if next_word != self.vocab.EOS_INDEX
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Add to completed sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            # Reduce beam length
            k -= len(complete_inds)

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]

            data = self.get_data_beam_search_t(data, prev_word_inds[incomplete_inds])
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            data["texts"] = next_word_inds[incomplete_inds].unsqueeze(1)

        if len(complete_seqs_scores) == 0:
            captions = torch.FloatTensor([0] * timesteps).unsqueeze(0)
        else:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            captions = torch.FloatTensor(complete_seqs[i]).unsqueeze(0)

        targets = sample_list.answers[:, 1:]
        sample_list.add_field("targets", targets)
        model_output = {"scores": outputs, "captions": captions}

        return model_output
