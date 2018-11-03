import os
import numpy as np

from torch.utils.data import ConcatDataset

from .utils import VocabDict
from pythia.core.constants import imdb_version
# TODO: Move in __all__ in __init__.py
from pythia.tasks.datasets.coco.coco_features_dataset \
    import COCOFeaturesDataset
from pythia.core.tasks.datasets.image_dataset import ImageDataset
from pythia.core.tasks.datasets.base_dataset import BaseDataset


def compute_answer_scores(answers, num_of_answers, unk_idx):
    scores = np.zeros((num_of_answers), np.float32)
    for answer in set(answers):
        if answer == unk_idx:
            scores[answer] = 0
        else:
            answer_count = answers.count(answer)
            scores[answer] = min(np.float32(answer_count)*0.3, 1)
    return scores


class VQA2Dataset(BaseDataset):
    def __init__(self, imdb_file,
                 image_feat_directories, verbose=False, **data_params):
        super(VQA2Dataset, self).__init__('vqa', data_params)

        if imdb_file.endswith('.npy'):
            imdb = ImageDataset(imdb_file)
        else:
            raise TypeError('unknown imdb format.')
        self.verbose = verbose
        self.imdb = imdb

        self.image_feat_directories = image_feat_directories
        self.data_params = data_params
        self.channel_first = data_params['image_depth_first']
        self.max_bboxes = (data_params['image_max_loc']
                           if 'image_max_loc' in data_params else None)

        self.max_valid_answer_length = 10

        # TODO: Remove after shifting to proper features standard
        self.first_element_idx = 1
        # TODO: Update T_encoder and T_decoder to proper names
        self.T_encoder = data_params['T_encoder']

        # TODO: Provide in the metadata itself
        self.load_answer = True
        # read the header of imdb file
        self.load_gt_layout = False
        data_version = self.imdb.get_version()

        if data_version != imdb_version:
            print("observed imdb_version is",
                  data_version,
                  "expected imdb version is",
                  imdb_version)
            raise TypeError('imdb version do not match.')

        if 'load_gt_layout' in data_params:
            self.load_gt_layout = data_params['load_gt_layout']

        self.data_root_dir = data_params['data_root_dir']
        vocab_answer_file = os.path.join(
            self.data_root_dir, data_params['vocab_answer_file'])

        # the answer dict is always loaded, regardless of self.load_answer
        self.answer_dict = VocabDict(vocab_answer_file)

        if self.load_gt_layout:
            self.T_decoder = data_params['T_decoder']
            self.assembler = data_params['assembler']
            self.prune_filter_module = (data_params['prune_filter_module']
                                        if 'prune_filter_module' in data_params
                                        else False)

        self.features_db = COCOFeaturesDataset(
                            image_feature_dirs=self.image_feat_directories,
                            channel_first=self.channel_first,
                            max_bboxes=self.max_bboxes,
                            imdb=self.imdb,
                            ndim=data_params.get('ndim', None),
                            dataset_type=data_params['dataset_type'],
                            fast_read=data_params['fast_read'],
                            return_info=self.config.get('return_info', False))

    def format_for_evalai(self, batch, answers):
        answers = answers.argmax(dim=1)

        predictions = []

        for idx, question_id in enumerate(batch['question_id']):
            answer = self.answer_dict.idx2word(answers[idx])

            predictions.append({
                'question_id': question_id.item(),
                'answer': answer
            })

        return predictions

    def __len__(self):
        return len(self.imdb) - 1

    def __getitem__(self, idx):
        input_seq = np.zeros((self.T_encoder), np.int32)
        idx += self.first_element_idx
        # TODO: Bring back commented out code when we reformat imdb
        # iminfo = self.imdb[idx]['info']
        iminfo = self.imdb[idx]
        seq_length = len(iminfo['question_tokens'])
        read_len = min(seq_length, self.T_encoder)
        tokens = iminfo['question_tokens'][:read_len]
        input_seq[:read_len] = ([self.text_vocab.stoi[w] for w in tokens])

        image_features = self.features_db[idx]

        answer_tokens = None

        valid_answers_idx = np.zeros((self.max_valid_answer_length), np.int32)
        valid_answers_idx.fill(-1)
        answer_scores = np.zeros(self.answer_dict.num_vocab, np.float32)
        if self.load_answer:
            if 'answer' in iminfo:
                answer_tokens = iminfo['answer']
            # if 'answer_tokens' in iminfo:
            #     answer_tokens = iminfo['answer_tokens']
            # elif 'valid_answers_tokens' in iminfo:
            #     valid_answers_tokens = iminfo['valid_answers_tokens']
            elif 'valid_answers' in iminfo:
                valid_answers_tokens = iminfo['valid_answers']
                if valid_answers_tokens[-1] == '<copy>':
                    valid_answers_tokens.pop()
                answer_tokens = np.random.choice(valid_answers_tokens)
                ans_idx = (
                    [self.answer_dict.word2idx(ans)
                     for ans in valid_answers_tokens])

                valid_answers_idx[:len(valid_answers_tokens)] = \
                    ans_idx
                answer_scores = (
                    compute_answer_scores(ans_idx,
                                          self.answer_dict.num_vocab,
                                          self.answer_dict.UNK_idx))

            answer_idx = self.answer_dict.word2idx(answer_tokens)

        if self.load_gt_layout:
            gt_layout_tokens = iminfo['gt_layout_tokens']
            if self.prune_filter_module:
                for n_t in range(len(gt_layout_tokens) - 1, 0, -1):
                    if (gt_layout_tokens[n_t - 1] in {'_Filter', '_Find'}
                            and gt_layout_tokens[n_t] == '_Filter'):
                        gt_layout_tokens[n_t] = None
                gt_layout_tokens = [t for t in gt_layout_tokens if t]
            gt_layout = np.array(self.assembler.module_list2tokens(
                gt_layout_tokens, self.T_decoder))

        sample = dict(texts=input_seq,
                      texts_lens=seq_length,
                      question_id=iminfo.get('question_id', None))

        for idx in range(len(image_features.keys())):
            if ("image_feature_%d" % idx) in image_features:
                image_feature = image_features["image_feature_%d" % idx]
                feat_key = "image_feature_%s" % str(idx)
                sample[feat_key] = image_feature
            else:
                break

        if "image_info_0" in image_features:
            info = image_features['image_info_0']
            if "max_bboxes" in info:
                sample['image_dim'] = info['max_bboxes']

            if "bboxes" in info:
                sample['image_boxes'] = info['bboxes']

        if self.load_answer:
            sample['answer_label'] = answer_idx
        if self.load_gt_layout:
            sample['gt_layout'] = gt_layout

        if valid_answers_idx is not None:
            sample['valid_ans_labels'] = valid_answers_idx
            sample['answers'] = answer_scores

        # used for error analysis and debug,
        # output question_id, image_id, question, answer,valid_answers,
        # if self.verbose:
        #     sample['verbose_info'] = iminfo

        return sample


class VQAConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(VQAConcatDataset, self).__init__(datasets)
        self.text_vocab = datasets[0].text_vocab

        if hasattr(datasets[0], 'context_vocab'):
            self.context_vocab = datasets[0].context_vocab

        self.name = self.datasets[0].name
        self.answer_dict = datasets[0].answer_dict

    def calculate_loss(self, output, expected_output, info):
        return self.datasets[0].calculate_loss(output, expected_output, info)

    def init_loss_and_metrics(self, config):
        self.datasets[0].init_loss_and_metrics(config)

    def report_metrics(self, loss=None, extra_info=None, should_print=True):
        self.datasets[0].report_metrics(loss, extra_info, should_print)

    def reset_meters(self):
        self.datasets[0].reset_meters()

    def prepare_batch(self, batch):
        return self.datasets[0].prepare_batch(batch)

    def format_for_evalai(self, batch, answers):
        return self.datasets[0].format_for_evalai(batch, answers)
