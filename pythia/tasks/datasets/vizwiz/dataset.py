import re
import os
import numpy as np

from collections import Counter
from torch.autograd import Variable

from pythia.tasks.datasets.vqa2.dataset import VQA2Dataset
from pythia.core.registry import registry
from pythia.core.text.utils import generate_ngrams_range
from pythia.tasks.datasets.vqa2.utils import word_tokenize


class VizWizDataset(VQA2Dataset):
    SPELL_CHECKER_INITIAL_CAPACITY = 83000
    MAX_EDIT_DISTANCE_DICTIONARY = 2
    SPELL_CHECKER_PREFIX_LENGTH = 7
    SPELL_CHECKER_TERM_INDEX = 0
    SPELL_CHECKER_COUNT_INDEX = 0

    def __init__(self, imdb_file, image_feat_directories, verbose=False,
                 **data_params):
        super(VizWizDataset, self).__init__(imdb_file, image_feat_directories,
                                            verbose, **data_params)
        self.data_params = data_params
        self.name = 'vizwiz'

        self.writer = registry.get('writer')
        self.sentence_split_regex = re.compile(r'(\s+)')
        self.use_spell_checker = False
        self.num_times_predicted = 0
        self.num_times_predicted_from_ocr = 0
        self.num_times_predicted_from_as = 0
        self.correct_predicted_from_ocr = 0
        self.correct_predicted_from_as = 0
        self.partial_predicted_from_ocr = 0

        if 'use_ocr' not in self.data_params:
            self.writer.write("use_ocr configuration not present. "
                              "Setting to False.", "warning")
            self.use_ocr = False
            self.data_params['use_ocr'] = False
        else:
            self.use_ocr = self.data_params['use_ocr']

        if 'copy_type' not in self.data_params:
            self.writer.write("copy_type configuration is not present. "
                              "Setting to False", "warning")
            self.copy_type = False
            self.data_params['copy_type'] = True
        else:
            self.copy_type = self.data_params['copy_type']

        if 'copy_included' not in self.data_params:
            self.writer.write("copy_included configuration not present. "
                              "Setting to False", 'warning')
            self.copy_included = False
            self.data_params['copy_included'] = False
        else:
            self.copy_included = self.data_params['copy_included']

        self.use_ngrams = self.data_params.get('use_ngrams', False)
        if self.use_ocr:
            self.context_max_len = self.config['context_max_len']
            self.context_seq_shape = (self.context_max_len)
            self.context_type = np.int32

            self.default_tokens = ["unanswerable" for _ in
                                   range(self.context_max_len)]

            if self.context_vocab.type == "model":
                self.context_seq_shape = (self.context_max_len,
                                          self.context_vocab.get_dim())
                self.context_type = np.float32

        if self.copy_included:
            self.max_valid_answer_length = 11

        if self.copy_type == "soft":
            self.writer.write("use_soft_copy is enabled, loading"
                              " sym spell checker")

            self.use_soft_copy = True

        if 'use_spell_checker' in self.data_params:
            self._init_spell_checker()

    def _init_spell_checker(self):
        from symspellpy.symspellpy import SymSpell

        if 'vocab_spell_checker' in self.data_params:
            file_path = self.data_params['vocab_spell_checker']
            vocab_path = file_path if os.path.isabs(file_path) \
                else os.path.join(self.data_root_dir, file_path)

            if not os.path.exists(vocab_path):
                self.writer.write("Vocab for spell checker, doesn't"
                                  " exist at %s" % vocab_path, 'error')
                return
            self.use_spell_checker = True

            self.spell_checker = SymSpell(self.SPELL_CHECKER_INITIAL_CAPACITY,
                                          self.MAX_EDIT_DISTANCE_DICTIONARY,
                                          self.SPELL_CHECKER_PREFIX_LENGTH)
            self.spell_checker.load_dictionary(vocab_path,
                                               self.SPELL_CHECKER_TERM_INDEX,
                                               self.SPELL_CHECKER_COUNT_INDEX)

    def prepare_batch(self, batch):
        data, obs = super(VizWizDataset, self).prepare_batch(batch)

        if 'order_vectors' in batch:
            order_vectors = batch['order_vectors']
            order_vectors = Variable(order_vectors, requires_grad=False,
                                     volatile=False)

            if self.use_cuda:
                order_vectors = order_vectors.cuda()

            data['info']['order_vectors'] = order_vectors

        return data, obs

    def __getitem__(self, idx):
        sample = super(VizWizDataset, self).__getitem__(idx)
        idx = self.first_element_idx + idx
        image = self.imdb[idx]
        sample['image_id'] = image['image_name']

        if self.use_ocr:
            context_seq = np.zeros(self.context_seq_shape, self.context_type)
            tokens = image['ocr_tokens']
            tokens = [word_tokenize(word) for word in tokens]
            original_len = min(len(tokens), self.context_max_len)
            if self.use_spell_checker:
                spell_checked = self.spell_checker.lookup_compound(
                    " ".join(tokens), 2
                )
                corrected_tokens = spell_checked[0].term.split(" ")
                tokens = corrected_tokens + tokens

            if self.use_ngrams:
                tokens = list(generate_ngrams_range(tokens, (1, 3)))

            attention_supervision_org = image.get('att_sup', None)
            # TODO: Uncomment or remove this if we finalize on using
            # classes + attributes in context supervision
            # final_tokens = []
            # is_ocr = image['image_bbox_source']
            #
            # for token, ocr in zip(tokens, is_ocr):
            #     if ocr.item() > 0:
            #         final_tokens.append(token)

            context_len = min(len(tokens), self.context_max_len)
            final_tokens = tokens[:context_len]

            token_idxs = [np.mean([self.context_vocab.stoi[w]
                                   for w in word.split(' ')], axis=0)
                          for word in final_tokens]

            order_vectors = np.zeros((self.context_max_len,
                                      self.context_max_len), np.float32)

            for x in range(len(final_tokens)):
                order_vectors[x][x] = 1

            if context_len > 0:
                context_seq[:context_len] = token_idxs

            sample['ocr_tokens'] = self.default_tokens[:]
            sample['ocr_tokens'][:context_len] = final_tokens
            # print(final_tokens)
            sample['contexts'] = context_seq
            # Context dim is actually 'length' of the final context
            sample['context_dim'] = context_len
            sample['order_vectors'] = order_vectors

            self._update_based_on_copy(idx, sample)

            if attention_supervision_org is not None:
                attention_supervision = np.zeros((self.context_max_len),
                                                 np.float)
                attention_supervision[:original_len] = \
                    attention_supervision_org[:original_len]
                sample['attention_supervision'] = attention_supervision
        return sample

    def _update_based_on_copy(self, idx, sample):
        if not self.copy_type:
            return
        iminfo = self.imdb[idx]
        context_len = sample['context_dim']

        scores = np.zeros((self.context_max_len), np.float32)
        answer_scores = sample['answers']

        if self.load_answer and 'all_answers' in iminfo:
            all_answers = iminfo['all_answers']

            if self.name == "textvqa":
                all_answers = all_answers[-6:]

            if all_answers[-1] == '<copy>':
                all_answers.pop()

            answer_counter = Counter()

            for token in all_answers:
                token = word_tokenize(token).strip()
                answer_counter[token] += 1
            # print("Possible OCR Answers")
            # Calculate scores for each of the OCR tokens and extend the answer
            # classification space
            for idx, token in enumerate(sample['ocr_tokens'][:context_len]):
                token = word_tokenize(token)
                answer_count = answer_counter[token]

                if answer_count == 0:
                    continue
                # print(idx + self.answer_space_size)
                scores[idx] = min(np.float32(answer_count) * 0.3, 1)
                answer_scores[self.answer_dict.UNK_idx] = 0

        answer_scores_expanded = np.zeros((
            self.answer_space_size + self.context_max_len), np.float32)
        answer_scores_expanded[:self.answer_space_size] = \
            answer_scores
        answer_scores_expanded[self.answer_space_size:] = scores
        sample['answers'] = answer_scores_expanded

    @classmethod
    def init_args(self, parser):
        parser.add_argument_group("VizWiz Top Down Bottom Up Model")
        parser.add_argument("--copy_mechanism", type=bool,
                            default=None,
                            help="If we should copy mechanism")

        parser.add_argument("--attention_supervision", type=bool,
                            default=None,
                            help="If attention supervision loss should"
                            " be used")

        return parser

    def format_for_evalai(self, batch, answers):
        answers = answers.argmax(dim=1)

        predictions = []
        for idx, image_id in enumerate(batch['image_id']):
            answer_id = answers[idx]

            if answer_id >= self.answer_space_size:
                answer_id -= self.answer_space_size
                answer = word_tokenize(batch['ocr_tokens'][answer_id][idx])
            else:
                answer = self.answer_dict.idx2word(answer_id)
            # 'COCO_vizwiz_test_000000020255' -> 'VizWiz_test_000000020255.jpg'
            predictions.append({
                'image': "_".join(["VizWiz"] + image_id.split("_")[2:])
                         + ".jpg",
                'answer': answer
            })

        return predictions
