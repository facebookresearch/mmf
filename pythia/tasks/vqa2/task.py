# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os

from torch.utils.data import ConcatDataset

from .dataset import VQA2Dataset
from pythia.tasks.task_base import BaseTask


class VQA2Task(BaseTask):
    def __init__(self, dataset_type):
        super(VQA2Task, self).__init__('VQA2', dataset_type)

    def load(self, **opts):
        image_features = opts['image_feat_train'][0].split(',')
        self.num_image_features = len(image_features)

        if self.dataset_type == 'train':
            self.dataset = prepare_train_data_set(**opts)
        elif self.dataset_type == 'dev':
            self.dataset = prepare_eval_data_set(**opts)
        elif self.dataset_type == 'test':
            self.dataset = prepare_test_data_set(**opts)
        else:
            raise NotImplementedError("Unknown dataset type: %s"
                                      % self.dataset_type)
        # TODO: Load here
        return

    def build(self, opts):
        # TODO: Build actually here
        return

    def update_config_for_model(self, config):
        config['num_vocab_txt'] = self.dataset.vocab_dict.num_vocab
        config['num_choices'] = self.dataset.answer_dict.num_vocab
        config['num_image_features'] = self.num_image_features

    def init_args(self, parser):
        parser.add_argument_group("VQA2 task specific arguments")
        parser.add_argument('--data_root_dir', type=str, default="../data",
                            help="Root directory for data")

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)


class VQAConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(VQAConcatDataset, self).__init__(datasets)
        self.vocab_dict = datasets[0].vocab_dict
        self.answer_dict = datasets[0].answer_dict


def prepare_data_set(imdb_file_label, image_dir_label, **data_config):
    # get the potential shared data_config info
    data_root_dir = data_config['data_root_dir']
    vocab_question_file = os.path.join(
        data_root_dir, data_config['vocab_question_file'])
    vocab_answer_file = os.path.join(
        data_root_dir, data_config['vocab_answer_file'])
    question_max_len = data_config['question_max_len'] \
        if 'question_max_len' in data_config else 26

    layout_max_len = 0
    if 'vocab_layout_file' in data_config:
        layout_max_len = data_config['layout_max_len'] \
            if 'layout_max_len' in data_config else 13

    prune_filter_module = data_config['prune_filter_module'] \
        if 'prune_filter_module' in data_config else False
    image_depth_first = data_config['image_depth_first']
    image_fast_reader = data_config['image_fast_reader'] \
        if 'image_fast_reader' in data_config else False
    verbose = data_config['verbose'] if 'verbose' in data_config else False
    test_mode = data_config['test_mode'] \
        if 'test_mode' in data_config else False

    imdb_files = data_config[imdb_file_label]
    image_feat_dirs = data_config[image_dir_label]
    assert len(imdb_files) == len(image_feat_dirs), \
        image_dir_label + "has different length with " + image_dir_label
    image_max_loc = data_config['image_max_loc'] \
        if 'image_max_loc' in data_config else None

    datasets = []
    for imdb_file_trn_name, image_feat_dir in zip(imdb_files, image_feat_dirs):
        imdb_file_trn = os.path.join(data_root_dir, imdb_file_trn_name)
        image_feat_dirs = [os.path.join(data_root_dir, d)
                           for d in image_feat_dir.split(',')]

        train_dataset = VQA2Dataset(imdb_file=imdb_file_trn,
                                    image_feat_directories=image_feat_dirs,
                                    T_encoder=question_max_len,
                                    T_decoder=layout_max_len,
                                    assembler=None,
                                    vocab_question_file=vocab_question_file,
                                    vocab_answer_file=vocab_answer_file,
                                    prune_filter_module=prune_filter_module,
                                    image_depth_first=image_depth_first,
                                    fastRead=image_fast_reader,
                                    verbose=verbose,
                                    test_mode=test_mode,
                                    image_max_loc=image_max_loc)
        datasets.append(train_dataset)

    dataset = VQAConcatDataset(datasets)

    return dataset


def prepare_train_data_set(**data_config):
    return prepare_data_set('imdb_file_train',
                            'image_feat_train',
                            **data_config)


def prepare_eval_data_set(**data_config):
    # TODO: Add enforce_slow_reader to task args
    enforce_slow_reader = data_config['enforce_slow_reader']
    if enforce_slow_reader:
        data_config['image_fast_reader'] = False

    return prepare_data_set('imdb_file_val', 'image_feat_val', **data_config)


def prepare_test_data_set(**data_config):
    data_config['image_fast_reader'] = False
    return prepare_data_set('imdb_file_test', 'image_feat_test', **data_config)
