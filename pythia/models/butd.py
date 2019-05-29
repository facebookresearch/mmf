# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from pythia.common.registry import registry
from pythia.modules.layers import ClassifierLayer
from pythia.models.pythia import Pythia


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

    def prepare_data(self, sample_list):
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
                (sample_list.get_batch_size(), 1),
                self.vocab.SOS_INDEX,
                dtype=torch.long,
            )
            data["decode_lengths"] = sample_list.answers.new_full(
                (sample_list.get_batch_size(), 1), self.text_processor.max_length
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
        if prev_output is not None and not self.teacher_forcing:
            output_softmax = torch.log_softmax(prev_output, dim=1)
            _, indices = torch.max(output_softmax, dim=1, keepdim=True)
            data["texts"] = torch.cat(
                (data["texts"], indices.view(batch_size_t, 1)), dim=1
            )

        if self.teacher_forcing:
            batch_size_t = sum([l > t for l in data["decode_lengths"]])
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
        data, sample_list, timesteps = self.prepare_data(sample_list)
        batch_size = sample_list.answers.size(0)
        scores = sample_list.answers.new_ones(
            (batch_size, self.text_processor.max_length, self.vocab_size),
            dtype=torch.float,
        )

        output = None
        batch_size_t = batch_size
        for t in range(timesteps):
            data, batch_size_t = self.get_data_t(t, data, batch_size_t, output)
            pi_t = data["texts"][:, t].unsqueeze(-1)
            embedding = self.word_embedding(pi_t)
            image_out, _ = self.process_feature_embedding(
                "image", sample_list, embedding[:, 0, :], batch_size_t=batch_size_t
            )
            output = self.classifier(image_out)
            scores[:batch_size_t, t] = output

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
