# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import socket

import torch


def compare_tensors(a, b):
    return torch.all(a.eq(b))


def dummy_args(model="cnn_lstm", dataset="clevr"):
    args = argparse.Namespace()
    args.opts = ["model={}".format(model), "dataset={}".format(dataset)]
    args.config_override = None
    return args


def is_network_reachable():
    try:
        # check if host name can be resolved
        host = socket.gethostbyname("one.one.one.one")
        # check if host is actually reachable
        s = socket.create_connection((host, 80), 2)
        s.close()
        return True
    except IOError as e:
        if e.errno == 101:
            pass
    return False
