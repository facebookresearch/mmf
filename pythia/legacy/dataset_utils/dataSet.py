# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os

import numpy as np
from torch.utils.data import Dataset

from dataset_utils import text_processing
from global_variables.global_variables import imdb_version


class faster_RCNN_feat_reader:
    def read(self, image_feat_path):
        return np.load(image_feat_path)


class CHW_feat_reader:
    def read(self, image_feat_path):
        feat = np.load(image_feat_path)
        assert feat.shape[0] == 1, "batch is not 1"
        feat = feat.squeeze(0)
        return feat


class dim_3_reader:
    def read(self, image_feat_path):
        tmp = np.load(image_feat_path)
        _, _, c_dim = tmp.shape
        image_feat = np.reshape(tmp, (-1, c_dim))
        return image_feat


class HWC_feat_reader:
    def read(self, image_feat_path):
        tmp = np.load(image_feat_path)
        assert tmp.shape[0] == 1, "batch is not 1"
        _, _, _, c_dim = tmp.shape
        image_feat = np.reshape(tmp, (-1, c_dim))
        return image_feat


class padded_faster_RCNN_feat_reader:
    def __init__(self, max_loc):
        self.max_loc = max_loc

    def read(self, image_feat_path):
        image_feat = np.load(image_feat_path)
        image_loc, image_dim = image_feat.shape
        tmp_image_feat = np.zeros((self.max_loc, image_dim), dtype=np.float32)
        tmp_image_feat[0:image_loc,] = image_feat
        image_feat = tmp_image_feat
        return (image_feat, image_loc)


class padded_faster_RCNN_with_bbox_feat_reader:
    def __init__(self, max_loc):
        self.max_loc = max_loc

    def read(self, image_feat_path):
        image_feat_bbox = np.load(image_feat_path)
        image_boxes = image_feat_bbox.item().get("image_bboxes")
        tmp_image_feat = image_feat_bbox.item().get("image_feat")
        image_loc, image_dim = tmp_image_feat.shape
        tmp_image_feat_2 = np.zeros((self.max_loc, image_dim), dtype=np.float32)
        tmp_image_feat_2[0:image_loc,] = tmp_image_feat
        tmp_image_box = np.zeros((self.max_loc, 4), dtype=np.int32)
        tmp_image_box[0:image_loc] = image_boxes

        return (tmp_image_feat_2, image_loc, tmp_image_box)


def parse_npz_img_feat(feat):
    return feat["x"]


def get_image_feat_reader(ndim, channel_first, image_feat, max_loc=None):
    if ndim == 2 or ndim == 0:
        if max_loc is None:
            return faster_RCNN_feat_reader()
        else:
            if isinstance(image_feat.item(0), dict):
                return padded_faster_RCNN_with_bbox_feat_reader(max_loc)
            else:
                return padded_faster_RCNN_feat_reader(max_loc)
    elif ndim == 3 and not channel_first:
        return dim_3_reader()
    elif ndim == 4 and channel_first:
        return CHW_feat_reader()
    elif ndim == 4 and not channel_first:
        return HWC_feat_reader()
    else:
        raise TypeError("unkown image feature format")


def compute_answer_scores(answers, num_of_answers, unk_idx):
    scores = np.zeros((num_of_answers), np.float32)
    for answer in set(answers):
        if answer == unk_idx:
            scores[answer] = 0
        else:
            answer_count = answers.count(answer)
            scores[answer] = min(np.float32(answer_count) * 0.3, 1)
    return scores


def read_in_image_feats(image_dirs, image_readers, image_file_name):
    image_feats = []
    for i, image_dir in enumerate(image_dirs):
        image_feat_path = os.path.join(image_dir, image_file_name)
        tmp_image_feat = image_readers[i].read(image_feat_path)
        image_feats.append(tmp_image_feat)

    return image_feats


"""
Note: here, the first element in this dataset is header,
including dataset_name, imdb_version, create_time, has_answer,has_gt_layout
"""


class vqa_dataset(Dataset):
    def __init__(self, imdb_file, image_feat_directories, verbose=False, **data_params):
        super(vqa_dataset, self).__init__()
        if imdb_file.endswith(".npy"):
            imdb = np.load(imdb_file)
        else:
            raise TypeError("unknown imdb format.")
        self.verbose = verbose
        self.imdb = imdb
        self.image_feat_directories = image_feat_directories
        self.data_params = data_params
        self.image_depth_first = data_params["image_depth_first"]
        self.image_max_loc = (
            data_params["image_max_loc"] if "image_max_loc" in data_params else None
        )
        self.vocab_dict = text_processing.VocabDict(data_params["vocab_question_file"])
        self.T_encoder = data_params["T_encoder"]

        # read the header of imdb file
        header_idx = 0
        self.first_element_idx = 1
        header = self.imdb[header_idx]
        self.load_answer = header["has_answer"]
        self.load_gt_layout = header["has_gt_layout"]
        self.load_gt_layout = False
        data_version = header["version"]
        if data_version != imdb_version:
            print(
                "observed imdb_version is",
                data_version,
                "expected imdb version is",
                imdb_version,
            )
            raise TypeError("imdb version do not match.")

        if "load_gt_layout" in data_params:
            self.load_gt_layout = data_params["load_gt_layout"]
        # the answer dict is always loaded, regardless of self.load_answer
        self.answer_dict = text_processing.VocabDict(data_params["vocab_answer_file"])

        if self.load_gt_layout:
            self.T_decoder = data_params["T_decoder"]
            self.assembler = data_params["assembler"]
            self.prune_filter_module = (
                data_params["prune_filter_module"]
                if "prune_filter_module" in data_params
                else False
            )
        else:
            print("imdb does not contain ground-truth layout")
            print("Loading model and config ...")

        # load one feature map to peek its size
        self.image_feat_readers = []
        for image_dir in self.image_feat_directories:
            image_file_name = os.path.basename(
                self.imdb[self.first_element_idx]["feature_path"]
            )
            image_feat_path = os.path.join(image_dir, image_file_name)
            feats = np.load(image_feat_path)
            self.image_feat_readers.append(
                get_image_feat_reader(
                    feats.ndim, self.image_depth_first, feats, self.image_max_loc
                )
            )

        self.fastRead = False
        self.testMode = False
        if data_params["test_mode"]:
            self.testMode = True
        if data_params["fastRead"]:
            self.fastRead = True
            self.featDict = {}
            image_count = 0
            image_dir0 = self.image_feat_directories[0]
            for feat_file in os.listdir(image_dir0):
                if feat_file.endswith("npy"):
                    image_feats = read_in_image_feats(
                        self.image_feat_directories, self.image_feat_readers, feat_file
                    )
                    self.featDict[feat_file] = image_feats
                    image_count += 1
            print("load %d images" % image_count)

    def __len__(self):
        if self.testMode:
            return 1
        else:
            return len(self.imdb) - 1

    def _get_image_features_(self, image_file_name):
        if self.fastRead:
            image_feats = self.featDict[image_file_name]
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

    def __getitem__(self, idx):
        input_seq = np.zeros((self.T_encoder), np.int32)
        idx += self.first_element_idx
        iminfo = self.imdb[idx]
        question_inds = [self.vocab_dict.word2idx(w) for w in iminfo["question_tokens"]]
        seq_length = len(question_inds)
        read_len = min(seq_length, self.T_encoder)
        input_seq[:read_len] = question_inds[:read_len]

        image_file_name = self.imdb[idx]["feature_path"]
        image_feats, image_boxes, image_loc = self._get_image_features_(image_file_name)

        answer = None
        valid_answers_idx = np.zeros((10), np.int32)
        valid_answers_idx.fill(-1)
        answer_scores = np.zeros(self.answer_dict.num_vocab, np.float32)
        if self.load_answer:
            if "answer" in iminfo:
                answer = iminfo["answer"]
            elif "valid_answers" in iminfo:
                valid_answers = iminfo["valid_answers"]
                answer = np.random.choice(valid_answers)
                valid_answers_idx[: len(valid_answers)] = [
                    self.answer_dict.word2idx(ans) for ans in valid_answers
                ]
                ans_idx = [self.answer_dict.word2idx(ans) for ans in valid_answers]
                answer_scores = compute_answer_scores(
                    ans_idx, self.answer_dict.num_vocab, self.answer_dict.UNK_idx
                )

            answer_idx = self.answer_dict.word2idx(answer)

        if self.load_gt_layout:
            gt_layout_tokens = iminfo["gt_layout_tokens"]
            if self.prune_filter_module:
                for n_t in range(len(gt_layout_tokens) - 1, 0, -1):
                    if (
                        gt_layout_tokens[n_t - 1] in {"_Filter", "_Find"}
                        and gt_layout_tokens[n_t] == "_Filter"
                    ):
                        gt_layout_tokens[n_t] = None
                gt_layout_tokens = [t for t in gt_layout_tokens if t]
            gt_layout = np.array(
                self.assembler.module_list2tokens(gt_layout_tokens, self.T_decoder)
            )

        sample = dict(input_seq_batch=input_seq, seq_length_batch=seq_length)

        for im_idx, image_feat in enumerate(image_feats):
            if im_idx == 0:
                sample["image_feat_batch"] = image_feat
            else:
                feat_key = "image_feat_batch_%s" % str(im_idx)
                sample[feat_key] = image_feat

        if image_loc is not None:
            sample["image_dim"] = image_loc

        if self.load_answer:
            sample["answer_label_batch"] = answer_idx
        if self.load_gt_layout:
            sample["gt_layout_batch"] = gt_layout

        if valid_answers_idx is not None:
            sample["valid_ans_label_batch"] = valid_answers_idx
            sample["ans_scores"] = answer_scores

        if image_boxes is not None:
            sample["image_boxes"] = image_boxes

        # used for error analysis and debug,
        # output question_id, image_id, question, answer,valid_answers,
        if self.verbose:
            sample["verbose_info"] = iminfo

        return sample
