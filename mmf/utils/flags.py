# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import sys

from mmf.common.registry import registry


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Core Arguments")
        self.parser.add_argument(
            "-co",
            "--config_override",
            type=str,
            default=None,
            help="Use to override config from command line directly",
        )
        self.parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="Modify config options from command line",
        )


flags = Flags()
