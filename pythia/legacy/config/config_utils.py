# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import demjson
import yaml

from config.collections import AttrDict
from config.function_config_lib import ModelParPair


def object_2_attributes(vals):
    if isinstance(vals, dict):
        if "type" in vals and "par" in vals and len(vals) == 2:
            v1 = ModelParPair(vals["type"])
            update_config(v1, vals["par"])
            return v1
        else:
            v2 = AttrDict()
            for key, value in vals.items():
                v2[key] = object_2_attributes(value)
            return v2
    elif isinstance(vals, list):
        v3 = []
        for val in vals:
            v3.append(object_2_attributes(val))
        return v3
    else:
        return vals


def update_config(orig, other):
    if isinstance(other, dict):
        if isinstance(orig, ModelParPair):
            if "method" not in other and "par" not in other:
                exit(
                    "could not update a model_par_pair when \
                     neither type or par exist"
                )
            else:
                if "method" in other:
                    orig.update_type(other["method"])
                if "par" in other:
                    update_config(orig["par"], other["par"])

        for key, value in other.items():
            if key not in orig:
                exit("unkown key:%s, in new config string " % key)
            if isinstance(value, dict) or isinstance(value, list):
                update_config(orig[key], value)
            else:
                orig[key] = value

    elif isinstance(other, list):
        if not isinstance(orig, list):
            raise TypeError("the updated value is not a list")

        elm_to_remove = []
        for i, other_i in enumerate(other):
            if i >= len(orig):
                orig.append(object_2_attributes(other_i))
            if other_i == ".":
                pass
            elif other_i == "-":
                elm_to_remove.append(i)
            elif isinstance(other_i, dict):
                update_config(orig[i], other_i)
            else:
                orig[i] = other_i

        if len(orig) > len(other):
            elm_to_remove += range(len(other), len(orig))

        final_attr = [i for j, i in enumerate(orig) if j not in elm_to_remove]
        orig.clear()
        orig += final_attr
    else:
        raise TypeError("unkown type of updated config")


# -----------------Merge configuration from config file---------------------- #


def __merge_config_from_file(cfg, file_path):
    with open(file_path, "r") as f:
        updates = yaml.load(f)
    update_config(cfg, updates)


def __merge_config_from_cmdline(cfg, cmdstring):
    updates = demjson.decode(cmdstring)
    update_config(cfg, updates)


def finalize_config(cfg, cfg_file_path, cfg_cmd_string):
    if cfg_file_path is not None:
        __merge_config_from_file(cfg, cfg_file_path)

    if cfg_cmd_string is not None:
        __merge_config_from_cmdline(cfg, cfg_cmd_string)

    cfg.immutable(True)


# ----------------extract config to simple dict------------------------------ #


def convert_cfg_to_dict(cfg):
    if isinstance(cfg, AttrDict):
        cfg_dict = {}
        for key, val in cfg.items():
            if isinstance(val, list):
                val_list = [convert_cfg_to_dict(x) for x in val]
                cfg_dict[key] = val_list
            else:
                cfg_dict[key] = convert_cfg_to_dict(val)
        return cfg_dict
    elif isinstance(cfg, ModelParPair):
        cfg_dict = {"method": cfg.method, "par": convert_cfg_to_dict(cfg.par)}
        return cfg_dict
    else:
        return cfg


# --------------------dump config ------------------------------------------- #


def dump_config(cfg, config_file):
    with open(config_file, "w") as outfile:
        yaml.dump(
            convert_cfg_to_dict(cfg), outfile, default_flow_style=False, encoding=None
        )
