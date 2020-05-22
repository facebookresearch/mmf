# Copyright (c) Facebook, Inc. and its affiliates.

import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers.modeling_bert import BertModel
from transformers.tokenization_auto import AutoTokenizer


class BertFeatExtractor:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).eval()
        self.model.cuda()

    def get_bert_embedding(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.Tensor([indexed_tokens]).long()
        segments_tensor = torch.Tensor([0] * len(tokenized_text)).long()
        with torch.no_grad():
            encoded_layers, _ = self.model(
                tokens_tensor.cuda(),
                segments_tensor.cuda(),
                output_all_encoded_layers=False,
            )
        return encoded_layers.squeeze()[0]


def extract_bert(imdb_path, out_path, group_id=0, n_groups=1):
    imdb = np.load(imdb_path)

    feat_extractor = BertFeatExtractor("bert-base-uncased")

    if group_id == 0:
        iterator_obj = tqdm(imdb[1:])
    else:
        iterator_obj = imdb[1:]

    for idx, el in enumerate(iterator_obj):
        if idx % n_groups != group_id:
            continue
        emb = feat_extractor.get_bert_embedding(el["question_str"])
        save_path = out_path + str(el["question_id"])
        np.save(save_path, emb.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imdb_path", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--group_id", type=int, default=0)
    parser.add_argument("--n_groups", type=int, default=1)
    args = parser.parse_args()
    extract_bert(args.imdb_path, args.out_path, args.group_id, args.n_groups)
