# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import glob
import math
import os
import re

import numpy as np
import yaml


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="job id start")
    parser.add_argument("--end", type=int, default=None, help="job id end")
    parser.add_argument("--out", type=str, required=True, help="out file name")
    parser.add_argument("--log_dir", type=str, help="directory for log", default="logs")

    args = parser.parse_args()
    args.end = args.start if args.end is None else args.end

    return args


JOBID_reg = re.compile(".*main.*_j(\d+\_\d+)_.*out")
EPOCH_ACC_REG = re.compile("i_epoch.*val_acc(?:\s+)?:(?:\s+)?(0\.\d+)")
BEST_EPOCH_ACC_REG = re.compile(
    "best_acc(?:\s+)?:(?:\s+)?(0\.\d+).*epoch(?:\s+)?:(?:\s+)?(\d+\/\d+)"
)


def parse_log(file):
    job_id = JOBID_reg.match(file).group(1)
    acc_s = []
    best_acc = 0
    best_epoch = ""
    out_dir = ""
    with open(file, "r") as f:
        for line in f:
            line = line.rstrip()
            if "snapshot_dir" in line:
                out_dir = line.split("=")[1]
            elif line.startswith("i_epoch"):
                acc = float(EPOCH_ACC_REG.match(line).group(1))
                acc_s.append(acc)
            elif "best_acc" in line:
                match = BEST_EPOCH_ACC_REG.match(line)
                best_acc = float(match.group(1))
                best_epoch = match.group(2)

    if best_acc == 0 and len(acc_s) > 0:
        best_acc = max(acc_s)
        best_epoch = str(acc_s.index(max(acc_s)) + 1) + "/" + str(len(acc_s))

    return job_id, best_acc, best_epoch, out_dir


"""
INFO TO PARSE FROM CONFIG
dataset
batch_size,
learning_rate,
eps
lr_step
lr_ratio
modal_combine:
      non_linear_elmt_multiply:
        hidden_size: 2048
        dropout: 0
model
    image_embedding_models:
      - top_down_attention:
          modal_combine:
            non_linear_elmt_multiply:
              hidden_size: 2048
              dropout: 0.2
          transform:
            linear:
              out_dim: 1
          normalization: softmax

"""


def parse_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f)

    feats = "|".join(config["data"]["image_feat_train"])
    imdb = "|".join(config["data"]["imdb_file_train"])
    eps = (
        config["optimizer"]["par"]["eps"] if "eps" in config["optimizer"]["par"] else 0
    )
    eps = str(eps)

    result = [
        feats,
        imdb,
        config["data"]["batch_size"],
        config["optimizer"]["par"]["lr"],
        eps,
        config["optimizer"]["par"]["weight_decay"],
        config["training_parameters"]["lr_step"],
        config["training_parameters"]["lr_ratio"],
        config["model"]["modal_combine"]["non_linear_elmt_multiply"]["hidden_size"],
        config["model"]["modal_combine"]["non_linear_elmt_multiply"]["dropout"],
    ]

    res = "\t".join(list(map(str, result)))
    return res


CFG_reg = re.compile(".*--config\s+(\S+)")


def extract_config_file_name(out_file):
    err_file = out_file.replace(".out", ".err")
    with open(err_file, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("+ EXTRA_ARGS="):
                cfg_file = CFG_reg.match(line).group(1)
    return cfg_file


def parse_a_job(log_dir, job_id):
    bests = []
    complete = False
    config_file = None
    config_file_name = ""
    for file in glob.glob(log_dir + "/main*j" + str(job_id) + "_*out", recursive=True):
        job_id_p, best_acc, best_epoch, out_dir = parse_log(file)
        if out_dir == "":
            continue

        if os.path.exists(os.path.join(out_dir, "config.yaml")) and config_file is None:
            config_file = os.path.join(out_dir, "config.yaml")
            config_file_name = extract_config_file_name(file)

        [e, t] = list(map(int, best_epoch.split("/")))
        if t > e:
            if not complete:
                bests = [best_acc]
                complete = True
            else:
                bests.append(best_acc)
        else:
            if not complete:
                bests.append(best_acc)

    mean = 0 if len(bests) == 0 else np.mean(bests)
    sd = 0 if len(bests) == 0 else math.sqrt(np.var(bests))
    max_val = 0 if len(bests) == 0 else np.max(bests)

    if config_file is None:
        return None
    cfg_info = parse_config(config_file)

    bests_str = "\t".join(["%0.4f" % i for i in bests])

    result = (
        str(job_id)
        + "\t"
        + config_file_name
        + "\t"
        + cfg_info
        + "\t"
        + "%.4f" % mean
        + "\t"
        + "%.4f" % sd
        + "\t"
        + "%.4f" % max_val
        + "\t"
        + str(len(bests))
        + "\t"
        + str(complete)
        + "\t"
        + config_file
        + "\t"
        + bests_str
    )
    return result


HEAD = "job_id\tconfig_file_name\ttrain_feats\ttrain_imdb\tbatch_size\tlr\teps\tweigt_decay\tlr_step\tlr_ratio\thid_sz\tdrop\tmean\tsd\tmax\trepeats\tcomplete\tconfig\tvals"


if __name__ == "__main__":
    args = parse_arg()
    out = args.out
    i = 0

    with open(out, "w") as w:
        w.write(HEAD + "\n")
        for md_idx in range(args.start, args.end + 1):
            files = glob.glob(
                args.log_dir + "/main*j" + str(md_idx) + "_*out", recursive=True
            )

            if files is None or len(files) == 0:
                continue

            if (i + 1) % 10 == 0:
                print("process %d logs" % (i + 1))

            result = parse_a_job(args.log_dir, md_idx)

            if result is not None:
                i += 1
                w.write(result + "\n")

    print("DONE... result= %s" % out)
