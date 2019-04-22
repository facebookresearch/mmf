# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from pythia.tasks.vqa2.dataset import (get_image_feat_reader,
                                       read_in_image_feats)
from pythia.utils.vocab import GloVeIntersectedVocab


class VisualDialogDataset(Dataset):
    def __init__(self, imdb_file, image_feat_directories, verbose=False, **data_params):
        super(VisualDialogDataset, self).__init__()

        imdb = None
        if imdb_file.endswith(".npy"):
            imdb = np.load(imdb_file)
        elif imdb_file.endswith(".json"):
            with open(imdb_file, "r") as f:
                imdb = json.load(f)
        else:
            raise TypeError("Unknown IMDB format")

        self.verbose = verbose
        self.imdb = imdb
        self.image_feat_directories = image_feat_directories
        self.data_params = data_params
        self.first_element_idx = 0
        self.image_depth_first = data_params["image_depth_first"]
        self.image_max_loc = (
            data_params["image_max_loc"] if "image_max_loc" in data_params else None
        )

        self.max_seq_len = data_params["max_seq_len"]
        self.max_history_len = data_params["max_history_len"]
        self.vocab = GloVeIntersectedVocab(
            data_params["vocab_file"], data_params["embedding_name"]
        )
        self.fast_read = False
        self.test_mode = False
        if data_params["test_mode"]:
            self.test_mode = True

        self._init_image_readers()
        self._try_fast_read()
        self._process_dialogues()

    def _try_fast_read(self):
        if self.data_params["fast_read"]:
            self.fast_read = True
            self.feat_dict = {}
            image_count = 0
            image_dir = self.image_feat_directories[0]
            for feat_file in os.listdir(image_dir):
                if feat_file.endswith("npy"):
                    image_feats = read_in_image_feats(
                        self.image_feat_directories, self.image_feat_readers, feat_file
                    )
                    self.feat_dict[feat_file] = image_feats
                    image_count += 1
            print("Loaded %d images" % image_count)

    def _init_image_readers(self):
        # load one feature map to peek its size
        self.image_feat_readers = []
        for image_dir in self.image_feat_directories:
            feature_path = self.imdb["dialogs"][self.first_element_idx]
            feature_path = feature_path["image_feature_path"]
            image_feat_path = os.path.join(image_dir, feature_path)
            feats = np.load(image_feat_path)
            self.image_feat_readers.append(
                get_image_feat_reader(
                    feats.ndim, self.image_depth_first, feats, self.image_max_loc
                )
            )

    def _get_image_features(self, image_file_name):
        if self.fast_read:
            image_feats = self.feat_dict[image_file_name]
            if len(image_feats) != len(self.image_feat_directories):
                exit(image_file_name + "have %d features" % len(image_feats))
        else:
            image_feats = read_in_image_feats(
                self.image_feat_directories, self.image_feat_readers, image_file_name
            )

        image_boxes = None
        image_loc = None

        if isinstance(image_feats[0], tuple):
            image_loc = image_feats[0][1]
            image_feats_return = [image_feats[0][0]] + image_feats[1:]
            if len(image_feats[0]) == 3:
                image_boxes = image_feats[0][2]
        else:
            image_feats_return = image_feats

        return image_feats_return, image_boxes, image_loc

    def _tokens_to_word_indices(self, tokens_list):
        stoi = self.vocab.get_stoi()

        if not isinstance(tokens_list, list):
            tokens_list = [tokens_list]

        for tokens in tokens_list:
            for idx, token in enumerate(tokens):
                tokens[idx] = stoi[token]

        return tokens_list

    def _process_dialogues(self):
        nthreads = self.__len__()
        nrounds = 10
        questions = torch.zeros(nthreads, nrounds, self.max_seq_len).long()
        question_lens = torch.zeros(nthreads, nrounds).long()
        history = torch.zeros(nthreads, nrounds, self.max_history_len).long()
        history_lens = torch.zeros(nthreads, nrounds).long()
        answer_options = torch.zeros(nthreads, nrounds, 100).long()

        question_tokens = self.imdb["questions"]
        question_tokens = self._tokens_to_word_indices(question_tokens)

        answer_tokens = self.imdb["answers"]
        answer_tokens = self._tokens_to_word_indices(answer_tokens)

        threads = self.imdb["dialogs"]
        gt_indices = torch.zeros(nthreads, nrounds).long()

        sos_index = self.vocab.SOS_INDEX
        eos_index = self.vocab.EOS_INDEX
        for idx, thread in enumerate(threads):
            # Start with caption as initial dialog history
            # Prepend with SOS and append with EOS to signal end of sequence
            previous_dialog = [sos_index]
            caption_tokens = self._tokens_to_word_indices(thread["caption"])
            previous_dialog += caption_tokens[0][: self.max_history_len]
            previous_dialog += [eos_index]

            for round_id, dialog in enumerate(thread["dialog"]):
                # Add question's tokens, length trimmed on max_seq_len
                question_id = dialog["question"]
                curr_question_tokens = torch.Tensor(question_tokens[question_id]).long()
                curr_question_tokens = curr_question_tokens[: self.max_seq_len]
                question_len = len(curr_question_tokens)
                question_lens[idx][round_id] = question_len
                questions[idx][round_id][:question_len] = curr_question_tokens

                gt_indices[idx][round_id] = dialog["gt_index"]
                answer_id = dialog["answer"]

                previous_dialog = previous_dialog[: self.max_history_len]
                prev_dialog_np = torch.Tensor(previous_dialog).long()
                history[idx][round_id][: len(previous_dialog)] = prev_dialog_np
                history_lens[idx][round_id] = len(previous_dialog)

                options = torch.LongTensor(dialog["answer_options"])
                answer_options[idx][round_id] = options.long()

                previous_dialog += [sos_index]
                previous_dialog += question_tokens[question_id]
                previous_dialog += [eos_index]

                previous_dialog += [sos_index]
                previous_dialog += answer_tokens[answer_id]
                previous_dialog += [eos_index]

                if len(previous_dialog) > self.max_history_len:
                    first_eos = previous_dialog.index(eos_index)
                    previous_dialog = previous_dialog[first_eos + 1 :]

        self.questions = questions
        self.question_lens = question_lens
        self.answer_options = answer_options
        self.answer_tokens = answer_tokens
        self.history = history
        self.history_lens = history_lens
        self.gt_indices = gt_indices

    def __len__(self):
        if self.test_mode:
            return 1
        else:
            return len(self.imdb["dialogs"])

    def __getitem__(self, idx):
        questions = self.questions[idx]
        histories = self.history[idx]
        questions_len = self.question_lens[idx]
        histories_len = self.history_lens[idx]
        gt_indices = self.gt_indices[idx]
        answer_options = self.answer_options[idx]

        answer_options_np = torch.zeros(
            questions.shape[0], len(answer_options[0]), self.max_seq_len
        ).long()
        answer_options_len = torch.zeros(
            questions.shape[0], len(answer_options[0])
        ).long()

        for idx, dialog_options in enumerate(answer_options):
            for option_id, option in enumerate(dialog_options):
                tokens = self.answer_tokens[option][: self.max_seq_len]
                answer_options_len[idx][option_id] = len(tokens)
                tokens = torch.LongTensor(tokens)
                answer_options_np[idx][option_id][: len(tokens)] = tokens

        expected_output = torch.zeros(questions.shape[0], len(answer_options[0])).long()

        for idx, gt_index in enumerate(gt_indices):
            expected_output[idx][gt_index.item()] = 1

        sample = {
            "questions": questions,
            "histories": histories,
            "questions_len": questions_len,
            "histories_len": histories_len,
            "gt_indices": gt_indices,
            "answer_options": answer_options_np,
            "answer_options_len": answer_options_len,
            "expected": expected_output,
        }

        image_file_name = self.imdb["dialogs"][idx]["image_feature_path"]
        image_feats, image_boxes, image_loc = self._get_image_features(image_file_name)

        n_questions = questions.size(0)
        for im_idx, image_feat in enumerate(image_feats):
            image_feat = torch.from_numpy(image_feat)
            feat_dims = len(image_feat.size())
            image_feat = image_feat.unsqueeze(0).repeat(n_questions, *([1] * feat_dims))
            if im_idx == 0:
                sample["image_feat_batch"] = image_feat
            else:
                feat_key = "image_feat_batch_%s" % str(im_idx)
                sample[feat_key] = image_feat

        if image_loc is not None:
            sample["image_dim"] = torch.LongTensor([image_loc] * n_questions)

        return sample

    def collate_fn(self, batch):
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        out = {}

        for key in merged_batch:
            out[key] = torch.stack(merged_batch[key], 0)

        return out
