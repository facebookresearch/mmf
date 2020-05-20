#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.

import sys

from mmf_cli.run import run


def predict():
    sys.argv.extend(["evaluation.predict=true"])
    run(predict=True)


if __name__ == "__main__":
    predict()
