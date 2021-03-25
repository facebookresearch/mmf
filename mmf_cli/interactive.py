#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import logging
import typing

from mmf.utils.configuration import _merge_with_dotlist
from mmf.utils.flags import flags
from mmf.utils.inference import Inference
from mmf.utils.logger import setup_logger
from omegaconf import OmegaConf


def construct_config(opts: typing.List[str]):
    config = OmegaConf.create({"checkpoint_path": ""})
    return _merge_with_dotlist(config, opts)


def interactive(opts: typing.Optional[typing.List[str]] = None):
    """Inference runs inference on an image and text provided by the user.
    You can optionally run inference programmatically by passing an optlist as opts.

    Args:
        opts (typing.Optional[typing.List[str]], optional): Optlist which can be used.
            to override opts programmatically. For e.g. if you pass
            opts = ["checkpoint_path=my/directory"], this will set the checkpoint.
    """
    if opts is None:
        parser = flags.get_parser()
        args = parser.parse_args()
    else:
        args = argparse.Namespace(config_override=None)
        args.opts = opts

    setup_logger()
    logger = logging.getLogger("mmf_cli.interactive")

    config = construct_config(args.opts)
    inference = Inference(checkpoint_path=config.checkpoint_path)
    logger.info("Enter 'exit' at any point to terminate.")
    logger.info("Enter an image URL:")
    image_url = input()
    logger.info("Got image URL.")
    logger.info("Enter text:")
    text = input()
    logger.info("Got text input.")
    while text != "exit":
        logger.info("Running inference on image and text input.")
        answer = inference.forward(image_url, {"text": text}, image_format="url")
        logger.info("Model response: " + answer)
        logger.info(
            f"Enter another image URL or leave it blank to continue using {image_url}:"
        )
        new_image_url = input()
        if new_image_url != "":
            image_url = new_image_url
        if new_image_url == "exit":
            break
        logger.info("Enter another text input:")
        text = input()


if __name__ == "__main__":
    interactive()
