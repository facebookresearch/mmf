# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest

import yaml

from pythia.tasks.processors import CaptionProcessor
from pythia.utils.configuration import ConfigNode


class TestTaskProcessors(unittest.TestCase):
    def test_caption_processor(self):
        path = os.path.join(
            os.path.abspath(__file__),
            "../../../pythia/common/defaults/configs/tasks/captioning/coco.yml",
        )
        with open(os.path.abspath(path)) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config = ConfigNode(config)
        captioning_config = config.task_attributes.captioning.dataset_attributes.coco
        caption_processor_config = captioning_config.processors.caption_processor
        vocab_path = os.path.join(os.path.abspath(__file__), "../../modules/vocab.txt")
        caption_processor_config.params.vocab.vocab_file = os.path.abspath(vocab_path)
        caption_processor = CaptionProcessor(caption_processor_config.params)

        tokens = [1, 4, 5, 6, 4, 7, 8, 2, 0, 0, 0]
        caption = caption_processor(tokens)

        # Test start, stop, pad are removed
        self.assertNotIn('<s>', caption["tokens"])
        self.assertNotIn('</s>', caption["tokens"])
        self.assertNotIn('<pad>', caption["tokens"])

        # Test caption is correct
        self.assertEqual(caption["caption"], "a man with a red helmet")
