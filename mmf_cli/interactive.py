#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.

import sys

from mmf_cli.run import run


def interactive():
    sys.argv.extend(["evaluation.predict=true"])
    print("hello world!")
    #run(predict=True)


if __name__ == "__main__":
    interactive()
