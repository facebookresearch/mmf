# Copyright (c) Facebook, Inc. and its affiliates.
"""
Entrypoint script used by TorchX to start the training run in each process
"""
from mmf_cli.fb_run import fb_scheduler_run


if __name__ == "__main__":
    fb_scheduler_run()
