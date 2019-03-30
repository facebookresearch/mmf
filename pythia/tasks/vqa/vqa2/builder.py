# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os

from .dataset import VQA2Dataset, VQAConcatDataset
from pythia.core.tasks.dataset_builder import DatasetBuilder
from pythia.core.registry import registry


@registry.register_builder('vqa2')
class VQA2Builder(DatasetBuilder):
    def __init__(self):
        super(VQA2Builder, self).__init__('VQA2')
        self.dataset_class = VQA2Dataset

    def _load(self, **opts):
        self.opts = opts
        dataset_type = opts['dataset_type']

        self.data_root_dir = opts['data_root_dir']
        image_features = opts['image_features']['train'][0].split(',')
        self.num_image_features = len(image_features)

        dataset = None
        if dataset_type == 'train':
            dataset = self.prepare_train_data_set(**opts)
        elif dataset_type == 'dev':
            dataset = self.prepare_eval_data_set(**opts)
        elif dataset_type == 'test':
            dataset = self.prepare_test_data_set(**opts)
        else:
            raise NotImplementedError("Unknown dataset type: %s"
                                      % dataset_type)

        self.dataset = dataset
        return dataset

    def _build(self, **opts):
        # TODO: Build actually here
        return

    def update_config_for_model(self, config):
        config['num_vocab_txt'] = self.dataset.text_vocab.get_size()
        config['vocab_size'] = self.dataset.text_vocab.get_size()
        config['num_choices'] = self.dataset.answer_dict.num_vocab
        config['num_image_features'] = self.num_image_features

    def init_args(self, parser):
        parser.add_argument_group("VQA2 task specific arguments")
        parser.add_argument('--data_root_dir', type=str, default="../data",
                            help="Root directory for data")
        parser.add_argument("-nfr", "--slow_read", type=bool,
                            default=None,
                            help="Disable fast read and "
                                 "load features on fly")

    def prepare_train_data_set(self, **data_config):
        return self.prepare_data_set('train',
                                     'train',
                                     **data_config)

    def prepare_eval_data_set(self, **data_config):
        # TODO: Add enforce_slow_reader to task args
        return self.prepare_data_set('val', 'val',
                                     **data_config)

    def prepare_test_data_set(self, **data_config):
        # NOTE: Don't fast load test
        data_config['slow_read'] = True
        return self.prepare_data_set('test', 'test',
                                     **data_config)

    def set_dataset_class(self, cls):
        self.dataset_class = cls

    def prepare_data_set(self, imdb_file_label,
                         image_dir_label, **data_config):
        # get the potential shared data_config info
        # TODO: Update this and move default stuff to configuration
        data_root_dir = data_config['data_root_dir']
        question_max_len = data_config.get('question_max_len', 26)

        layout_max_len = 0
        if 'vocab_layout_file' in data_config:
            layout_max_len = data_config.get('layout_max_len', 13)

        prune_filter_mod = data_config.get('prune_filter_module', False)
        fast_read = not data_config.get('slow_read', False)

        verbose = data_config.get('verbose', False)
        test_mode = data_config.get('test_mode', False)

        imdb_files = data_config['imdb_files'][imdb_file_label]
        image_feat_dirs = data_config['image_features'][image_dir_label]
        assert len(imdb_files) == len(image_feat_dirs), \
            image_dir_label + "has different length with " + image_dir_label

        datasets = []

        for imdb_file_trn_name, image_feat_dir in \
                zip(imdb_files, image_feat_dirs):

            if not os.path.isabs(imdb_file_trn_name):
                imdb_file_trn = os.path.join(data_root_dir, imdb_file_trn_name)
            else:
                imdb_file_trn = imdb_file_trn_name

            feat_dirs = [os.path.join(data_root_dir, d)
                         if not os.path.isabs(d) else d
                         for d in image_feat_dir.split(',')]

            cls = self.dataset_class
            train_dataset = cls(imdb_file=imdb_file_trn,
                                image_feat_directories=feat_dirs,
                                T_encoder=question_max_len,
                                T_decoder=layout_max_len,
                                assembler=None,
                                prune_filter_module=prune_filter_mod,
                                fast_read=fast_read,
                                verbose=verbose,
                                test_mode=test_mode,
                                **data_config)
            train_dataset.try_fast_read()
            datasets.append(train_dataset)

        dataset = VQAConcatDataset(datasets)

        return dataset
