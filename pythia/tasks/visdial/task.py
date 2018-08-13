import os

from torch.utils.data import ConcatDataset

from pythia.tasks.vqa2.task import VQA2Task
from .dataset import VisualDialogDataset


class VisualDialogTask(VQA2Task):
    def __init__(self, dataset_type):
        super(VisualDialogTask, self).__init__(dataset_type)
        self.task_name = 'VisualDialog'

    def prepare_data_set(self, imdb_file_label, image_feature_dir_label,
                         **data_config):
        data_root_dir = data_config['data_root_dir']

        vocab_file = os.path.join(data_root_dir, data_config['vocab_file'])
        embedding_name = data_config['embedding_name']
        max_seq_len = data_config['max_seq_len']
        max_history_len = data_config['max_history_len']
        image_depth_first = data_config['image_depth_first']
        image_fast_reader = data_config['image_fast_reader']

        if 'verbose' in data_config:
            verbose = data_config['verbose']
        else:
            verbose = False

        if 'test_mode' in data_config:
            test_mode = data_config['test_mode']
        else:
            test_mode = False

        if 'image_max_loc' in data_config:
            image_max_loc = data_config['image_max_loc']
        else:
            image_max_loc = False

        imdb_files = data_config[imdb_file_label]
        image_feat_dirs = data_config[image_feature_dir_label]

        condition = len(imdb_files) == len(image_feat_dirs)
        error = imdb_file_label + "length != " + image_feature_dir_label
        error += "length"
        assert condition, error

        datasets = []

        for imdb_file, image_feature_dir in \
                zip(imdb_files, image_feat_dirs):
            imdb_file = os.path.join(data_root_dir, imdb_file)
            image_feat_dirs = [os.path.join(data_root_dir, d)
                               for d in image_feature_dir.split(',')]
            args = {
                'imdb_file': imdb_file,
                'image_feat_directories': image_feat_dirs,
                'max_seq_len': max_seq_len,
                'max_history_len': max_history_len,
                'vocab_file': vocab_file,
                'image_depth_first': image_depth_first,
                'fast_read': image_fast_reader,
                'verbose': verbose,
                'test_mode': test_mode,
                'image_max_loc': image_max_loc,
                'embedding_name': embedding_name
            }

            train_dataset = VisualDialogDataset(**args)
            datasets.append(train_dataset)

        return VisualDialogConcatDataset(datasets)

    def prepare_batch(self, batch, use_cuda):
        return batch, None

    def update_config_for_model(self, config):
        config['num_vocab_txt'] = self.dataset.vocab.get_size()
        config['vocab_size'] = self.dataset.vocab.get_size()
        config['num_image_features'] = self.num_image_features


class VisualDialogConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(VisualDialogConcatDataset, self).__init__(datasets)
        self.vocab = datasets[0].vocab
