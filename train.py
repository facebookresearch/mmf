import timeit
import argparse
import random
import os
import demjson
import yaml
import logging
import datetime
from torch import optim
from torch.utils.data import DataLoader
from config.config_utils import finalize_config, dump_config
from config.config import cfg
from global_variables.global_variables import use_cuda
from train_model.dataset_utils import prepare_train_data_set, prepare_eval_data_set, prepare_test_data_set
from train_model.helper import build_model, run_model, print_result
from train_model.Loss import get_loss_criterion
from train_model.Engineer import one_stage_train
import glob
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from bisect import bisect

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="config yaml file")
    parser.add_argument("--out_dir", type=str, default=None, help="output directory, default is current directory")
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--config_overwrite', type=str, help="a json string to update yaml config file", default=None)
    parser.add_argument("--force_restart", action='store_true',
                        help="flag to force clean previous result and restart training")

    arguments = parser.parse_args()
    return arguments


def process_config(config_file, config_string):
    finalize_config(cfg, config_file, config_string)


def get_output_folder_name(config_basename, cfg_overwrite_obj, seed):
    m_name, _ = os.path.splitext(config_basename)

    if cfg_overwrite_obj is not None:
        f_name = yaml.safe_dump(cfg_overwrite_obj, default_flow_style=False)
        f_name = f_name.replace(':', '.').replace('\n', ' ').replace('/', '_')
        f_name = ' '.join(f_name.split())
        f_name = f_name.replace('. ', '.').replace(' ', '_')
        f_name += '_%d' % seed
    else:
        f_name = '%d' % seed

    return m_name, f_name

def lr_lambda_fun(i_iter):
    if i_iter <= cfg.training_parameters.wu_iters:
        alpha = float(i_iter) / float(cfg.training_parameters.wu_iters)
        return cfg.training_parameters.wu_factor * (1. - alpha) + alpha
    else:
        idx = bisect(cfg.training_parameters.lr_steps, i_iter)
        return pow(cfg.training_parameters.lr_ratio, idx)


def get_optim_scheduler(optimizer):
    return LambdaLR(optimizer, lr_lambda=lr_lambda_fun)


def print_eval(prepare_data_fun, out_label):
    model_file = os.path.join(snapshot_dir, "best_model.pth")
    pkl_res_file = os.path.join(snapshot_dir, "best_model_predict_%s.pkl" % out_label)
    out_file = os.path.join(snapshot_dir, "best_model_predict_%s.json" % out_label)

    data_set_test = prepare_data_fun(**cfg['data'], **cfg['model'], verbose=False)
    data_reader_test = DataLoader(data_set_test, shuffle=False,
                                  batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    ans_dic = data_set_test.answer_dict

    model = build_model(cfg, data_set_test)
    model.load_state_dict(torch.load(model_file)['state_dict'])

    question_ids, soft_max_result = run_model(model, data_reader_test, ans_dic.UNK_idx)
    print_result(question_ids, soft_max_result, ans_dic, out_file, json_only=False, pkl_res_file=pkl_res_file)


if __name__ == '__main__':
    start = timeit.default_timer()

    args = parse_args()
    config_file = args.config
    seed = args.seed if args.seed is not None else random.randint(1, 100000)
    process_config(config_file, args.config_overwrite)

    basename = 'default' if args.config is None else os.path.basename(args.config)

    cmd_cfg_obj = demjson.decode(args.config_overwrite) if args.config_overwrite is not None else None

    middle_name, final_name = get_output_folder_name(basename, cmd_cfg_obj, seed)

    out_dir = args.out_dir if args.out_dir is not None else os.getcwd()

    snapshot_dir = os.path.join(out_dir, "results", middle_name, final_name)
    boards_dir = os.path.join(out_dir, "boards", middle_name, final_name)

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    if not os.path.exists(boards_dir):
        os.makedirs(boards_dir)

    print("snapshot_dir=" + snapshot_dir)
    print("fast data reader = " + str(cfg['data']['image_fast_reader']))
    print("use cuda = " + str(use_cuda))

    logger = logging.getLogger('genie')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_").replace(":", "_").replace("-","_")
    hdlr = logging.FileHandler(os.path.join(snapshot_dir, 'genie_{}.log'.format(ts)))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)

    # dump the config file to snap_shot_dir
    config_to_write = os.path.join(snapshot_dir, "config.yaml")
    dump_config(cfg, config_to_write)

    train_dataSet = prepare_train_data_set(**cfg['data'], **cfg['model'])

    my_model = build_model(cfg, train_dataSet)
    state = my_model.state_dict()

    model = my_model
    if hasattr(my_model, 'module'):
        model = my_model.module

    params = [{'params': model.image_embedding_models_list.parameters(),
               'lr': cfg.optimizer.par.lr},
              {'params': model.question_embedding_models.parameters(),
               'lr': cfg.optimizer.par.lr},
              {'params': model.multi_modal_combine.parameters(),
               'lr': cfg.optimizer.par.lr},
              {'params': model.image_feature_encode_list.parameters(),
               'lr': cfg.optimizer.par.lr}]

    params += [{'params': model.classifier.parameters(),
                'lr': cfg.optimizer.par.lr}]

    if cfg['model']['failure_predictor']['hidden_1'] > 0:
        params += [{'params': model.failure_predictor.parameters(),
                    'lr': cfg.training_parameters.fp_lr}]
    if cfg['model']['question_consistency']['hidden_size'] > 0:
        params += [{'params': model.question_consistency.parameters(),
                    'lr': cfg.training_parameters.qc_lr}]

    ms = [m for m in model.modules()]
    pg = {m.__class__: list(m.parameters()) for m in ms}
    dq = {k: pg[k] for k in pg.keys() if len(pg[k]) > 0}
    dq = {k: dq[k] for k in dq.keys() if dq[k][0].requires_grad}

    print("PRINTING ALL TRAINABLE CLASSES")
    for k in dq.keys():
        print(k)

    my_optim = getattr(optim, cfg.optimizer.method)(params,
                                                    **cfg.optimizer.par)

    i_epoch = 0
    i_iter = 0
    if not args.force_restart:
        md_pths = os.path.join(snapshot_dir, "model_*.pth")
        files = glob.glob(md_pths)
        if len(files) > 0:
            latest_file = max(files, key=os.path.getctime)
            info = torch.load(latest_file)
            i_epoch = info['epoch']
            i_iter = info['iter']
            sd = info['state_dict']
            op_sd = info['optimizer']
            my_model.load_state_dict(sd)
            my_optim.load_state_dict(op_sd)

    scheduler = get_optim_scheduler(my_optim)

    my_loss = get_loss_criterion(cfg.loss)

    data_set_val = prepare_eval_data_set(**cfg['data'], **cfg['model'])

    data_reader_trn = DataLoader(dataset=train_dataSet, batch_size=cfg.data.batch_size, shuffle=True,
                                 num_workers=cfg.data.num_workers, drop_last=True)

    if cfg['model']['question_consistency']['hidden_size'] > 0:
        val_bs = 64
    else:
        val_bs = cfg.data.batch_size

    data_reader_val = DataLoader(data_set_val, shuffle=False, batch_size=val_bs,
                                 num_workers=cfg.data.num_workers)

    print("BEGIN TRAINING MODEL...")

    one_stage_train(my_model, data_reader_trn, my_optim, my_loss, data_reader_eval=data_reader_val,
                    snapshot_dir=snapshot_dir, log_dir=boards_dir, start_epoch=i_epoch, i_iter=i_iter,
                    scheduler=scheduler)

    print("BEGIN PREDICTING ON TEST/VAL set...")

    if 'predict' in cfg.run:
        print_eval(prepare_test_data_set, "test")
    if cfg.run == 'train+val':
        print_eval(prepare_eval_data_set, "val")

    end = timeit.default_timer()
    total_time = (end - start) / 3600.0

    print("total runtime(h): %.2f" % total_time)
