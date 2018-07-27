# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
from torch.utils.data import DataLoader
from train_model.dataset_utils import prepare_test_data_set,prepare_eval_data_set
import torch
from train_model.helper import run_model, print_result, build_model
from config.config_utils import finalize_config
from config.config import cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config yaml file")
    parser.add_argument("--out_prefix", type=str, required=True,
                        help="output file name prefix, will append .json or .pkl")
    parser.add_argument("--model_path", type=str, help="path of model", required=True)
    parser.add_argument("--batch_size", type=int,
                        help="batch_size for test, o.w. using the one in config file", default=None)
    parser.add_argument("--num_workers",type=int, help="num_workers in dataLoader, default 0", default=5)
    parser.add_argument("--json_only", action='store_true', help="flag for only need json result")
    parser.add_argument("--use_val",action='store_true',help="flag for using val data for test")

    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':

    args = parse_args()

    config_file = args.config
    out_file = args.out_prefix+".json"
    model_file = args.model_path

    finalize_config(cfg, config_file, None)

    batch_size = cfg['data']['batch_size'] if args.batch_size is None else args.batch_size
    if args.use_val:
        data_set_test = prepare_eval_data_set(**cfg['data'], **cfg['model'], verbose=True)
    else:
        data_set_test = prepare_test_data_set(**cfg['data'], **cfg['model'], verbose=True)
    data_reader_test = DataLoader(data_set_test, shuffle=False, batch_size=batch_size, num_workers=args.num_workers)
    ans_dic = data_set_test.answer_dict

    myModel = build_model(cfg, data_set_test)
    myModel.load_state_dict(torch.load(model_file)['state_dict'])

    myModel.eval()

    question_ids, soft_max_result = run_model(myModel, data_reader_test, ans_dic.UNK_idx)

    pkl_res_file = args.out_prefix + ".pkl" if not args.json_only else None

    print_result(question_ids, soft_max_result, ans_dic, out_file, args.json_only, pkl_res_file)



