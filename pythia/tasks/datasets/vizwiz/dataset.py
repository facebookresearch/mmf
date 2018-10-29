import numpy as np

from pythia.tasks.datasets.vqa2.dataset import VQA2Dataset
from pythia.core.registry import Registry


class VizWizDataset(VQA2Dataset):
    def __init__(self, imdb_file, image_feat_directories, verbose=False,
                 **data_params):
        super(VizWizDataset, self).__init__(imdb_file, image_feat_directories,
                                            verbose, **data_params)
        self.data_params = data_params
        self.name = 'vizwiz'

        self.writer = Registry.get('writer')

        if 'use_ocr' not in self.data_params:
            self.writer.write("use_ocr configuration not present. "
                              "Setting to False", 'warning')
            self.use_ocr = False
            self.data_params['use_ocr'] = False
        else:
            self.use_ocr = self.data_params['use_ocr']

        if 'copy_included' not in self.data_params:
            self.writer.write("copy_included configuration not present. "
                              "Setting to False", 'warning')
            self.copy_included = False
            self.data_params['copy_included'] = False
        else:
            self.copy_included = self.data_params['copy_included']

        if self.use_ocr:
            self.context_max_len = self.config['context_max_len']

        if self.copy_included:
            self.max_valid_answer_length = 11

    def __getitem__(self, idx):
        sample = super(VizWizDataset, self).__getitem__(idx)
        idx = self.first_element_idx + idx
        image = self.imdb[idx]
        sample['image_id'] = image['image_name']

        if self.use_ocr:
            context_seq = np.zeros((self.context_max_len), np.int32)
            tokens = image['ocr_tokens']
            attention_supervision = image.get('att_sup', None)

            # TODO: Uncomment or remove this if we finalize on using
            # classes + attributes in context supervision
            # final_tokens = []
            # is_ocr = image['image_bbox_source']
            #
            # for token, ocr in zip(tokens, is_ocr):
            #     if ocr.item() > 0:
            #         final_tokens.append(token)

            final_tokens = tokens
            token_idxs = [self.context_vocab.stoi[w] for w in final_tokens]
            context_len = min(len(token_idxs), self.context_max_len)
            context_seq[:context_len] = token_idxs[:context_len]

            sample['contexts'] = context_seq
            # Context dim is actually 'length' of the final context
            sample['context_dim'] = context_len
            if attention_supervision is not None:
                sample['attention_supervision'] = attention_supervision
        return sample

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
            answer = self.answer_dict.idx2word(answers[idx])
            # 'COCO_vizwiz_test_000000020255' -> 'VizWiz_test_000000020255.jpg'
            predictions.append({
                'image': "_".join(["VizWiz"] + image_id.split("_")[2:])
                         + ".jpg",
                'answer': answer
            })

        return predictions
