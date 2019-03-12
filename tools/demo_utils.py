import os
from pythia.dataset_utils import text_processing

from pythia.train_model import model_factory

import torch
from collections import OrderedDict

import numpy as np
import PIL

from pythia.train_model.Engineer import one_stage_run_model, masked_unk_softmax


def load_vocab_dict(path):
    return text_processing.VocabDict(path)


def create_vqa_model(config):
    vocab_dict_path = os.path.join(config['root_dir'], 'data',
                                   config['data']['vocab_question_file'])
    answer_dict_path = os.path.join(config['root_dir'], 'data',
                                    config['data']['vocab_answer_file'])
    vocab_dict = text_processing.VocabDict(vocab_dict_path)
    ans_dict = text_processing.VocabDict(answer_dict_path)
    image_feats = config['data']['image_feat_train'][0].split(',')
    model_config = config['model']

    num_image_feat = len(image_feats)
    num_vocab_txt = vocab_dict.num_vocab
    num_answers = ans_dict.num_vocab
    model = model_factory.prepare_model(num_vocab_txt, num_answers, num_image_feat=num_image_feat,
                                        **model_config)
    return model


def load_pretrained_model(model, filepath):
    state_dict = torch.load(filepath)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k not in model.state_dict().keys():
            name = k.replace('module.', '')
        else:
            name = k
        if v.shape != model.state_dict()[name].shape:
            v = v.reshape(model.state_dict()[name].shape)
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    return model

def preprocess_text(question, vocab_dict, question_max_len=14):

    question_tokens = text_processing.tokenize(question)
    input_seq = np.zeros((question_max_len), np.int32)
    question_inds = (
        [vocab_dict.word2idx(w) for w in question_tokens])
    seq_length = len(question_inds)
    read_len = min(seq_length, question_max_len)
    input_seq[:read_len] = question_inds[:read_len]
    sample = dict(input_seq_batch=torch.from_numpy(input_seq).unsqueeze(0),
                  seq_length_batch=torch.Tensor((seq_length,)).unsqueeze(0))
    return sample

def extract_resnet_feat(model, image):
    img_feat = model(image)
    img_feat = img_feat.permute(0, 2, 3, 1).contiguous().view(196,-1)
    return img_feat


def preprocess_image(im_file, resnet_model):
    sample = {}

    image_feats = []
    # append detectron feature
    image_feats.append(torch.rand(100, 2048))
    # append resnet feature
    image_feats.append(extract_resnet_feat(resnet_model, im_file))

    for im_idx, image_feat in enumerate(image_feats):
        if im_idx == 0:
            sample['image_feat_batch'] = image_feat.unsqueeze(0)
        else:
            feat_key = "image_feat_batch_%s" % str(im_idx)
            sample[feat_key] = image_feat.unsqueeze(0)

    sample['image_dim'] = torch.Tensor((100,))
    return sample

def load_image(filepath):
    return PIL.Image.open(filepath).convert('RGB')


def evaluate_sample(model, resnet_model,
                    vocab_dict, image, question, UNK_idx=0):
    # preprocess
    preproc_text = preprocess_text(question, vocab_dict)
    preproc_image = preprocess_image(image, resnet_model)
    sample = {**preproc_text, **preproc_image}
        #
    logit_res = one_stage_run_model(sample, model, eval_mode=True)
    softmax_res = masked_unk_softmax(logit_res, dim=1, mask_idx=UNK_idx)
    softmax_res = softmax_res.data.cpu().numpy().astype(np.float16)
    return softmax_res

def get_logits(model, resnet_model,
                    vocab_dict, image, question, UNK_idx=0):
    preproc_text = preprocess_text(question, vocab_dict)
    preproc_image = preprocess_image(image, resnet_model)
    sample = {**preproc_text, **preproc_image}
    logit_res = one_stage_run_model(sample, model, eval_mode=True)
    return logit_res


def get_top_n_results(softmax_result, vocab, n=1):
    answer_idx = softmax_result[0].argsort()[-n:]
    return [vocab.idx2word(idx) for idx in answer_idx]


def get_classes_and_scores(softmax_result, vocab, n=1):
    answer_idx = softmax_result[0].argsort()[-n:]
    classes = [vocab.idx2word(idx) for idx in answer_idx]
    scores = sorted(softmax_result[0])[-n:]

    return classes, scores
