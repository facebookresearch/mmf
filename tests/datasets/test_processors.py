# Copyright (c) Facebook, Inc. and its affiliates.
import os
import tempfile
import unittest

import torch
from mmf.datasets.processors.processors import (
    CaptionProcessor,
    EvalAIAnswerProcessor,
    MultiClassFromFile,
    MultiHotAnswerFromVocabProcessor,
    TransformerBboxProcessor,
)
from mmf.utils.configuration import load_yaml
from omegaconf import OmegaConf

from ..test_utils import compare_tensors


class TestDatasetProcessors(unittest.TestCase):
    def _get_config(self, path):
        path = os.path.join(os.path.abspath(__file__), path)
        config = load_yaml(os.path.abspath(path))
        return config

    def test_caption_processor(self):
        config = self._get_config("../../../mmf/configs/datasets/coco/defaults.yaml")
        captioning_config = config.dataset_config.coco
        caption_processor_config = captioning_config.processors.caption_processor

        vocab_path = os.path.join(
            os.path.abspath(__file__), "..", "..", "data", "vocab.txt"
        )
        caption_processor_config.params.vocab.type = "random"
        caption_processor_config.params.vocab.vocab_file = os.path.abspath(vocab_path)
        caption_processor = CaptionProcessor(caption_processor_config.params)

        tokens = [1, 4, 5, 6, 4, 7, 8, 2, 0, 0, 0]
        caption = caption_processor(tokens)

        # Test start, stop, pad are removed
        self.assertNotIn("<s>", caption["tokens"])
        self.assertNotIn("</s>", caption["tokens"])
        self.assertNotIn("<pad>", caption["tokens"])

        # Test caption is correct
        self.assertEqual(caption["caption"], "a man with a red helmet")

    def test_multi_hot_answer_from_vocab_processor(self):
        config = self._get_config("../../../mmf/configs/datasets/clevr/defaults.yaml")
        clevr_config = config.dataset_config.clevr
        answer_processor_config = clevr_config.processors.answer_processor

        # Test num_answers==1 case
        vocab_path = os.path.join(
            os.path.abspath(__file__), "..", "..", "data", "vocab.txt"
        )
        answer_processor_config.params.vocab_file = os.path.abspath(vocab_path)
        answer_processor = MultiHotAnswerFromVocabProcessor(
            answer_processor_config.params
        )
        processed = answer_processor({"answers": ["helmet"]})
        answers_indices = processed["answers_indices"]
        answers_scores = processed["answers_scores"]

        self.assertTrue(
            compare_tensors(answers_indices, torch.tensor([5] * 10, dtype=torch.long))
        )
        expected_answers_scores = torch.zeros(19, dtype=torch.float)
        expected_answers_scores[5] = 1.0
        self.assertTrue(compare_tensors(answers_scores, expected_answers_scores))

        # Test multihot when num answers greater than 1
        answer_processor_config.params.vocab_file = os.path.abspath(vocab_path)
        answer_processor_config.params.num_answers = 3
        answer_processor = MultiHotAnswerFromVocabProcessor(
            answer_processor_config.params
        )
        processed = answer_processor({"answers": ["man", "with", "countryside"]})
        answers_indices = processed["answers_indices"]
        answers_scores = processed["answers_scores"]
        self.assertTrue(
            compare_tensors(
                answers_indices,
                torch.tensor([2, 3, 15, 2, 3, 15, 2, 3, 15, 2], dtype=torch.long),
            )
        )
        expected_answers_scores = torch.zeros(19, dtype=torch.float)
        expected_answers_scores[2] = 1.0
        expected_answers_scores[3] = 1.0
        expected_answers_scores[15] = 1.0
        self.assertTrue(compare_tensors(answers_scores, expected_answers_scores))

        # Test unk
        processed = answer_processor({"answers": ["test", "answer", "man"]})
        answers_indices = processed["answers_indices"]
        answers_scores = processed["answers_scores"]
        self.assertTrue(
            compare_tensors(
                answers_indices,
                torch.tensor([0, 0, 2, 0, 0, 2, 0, 0, 2, 0], dtype=torch.long),
            )
        )
        expected_answers_scores = torch.zeros(19, dtype=torch.float)
        expected_answers_scores[2] = 1.0
        self.assertTrue(compare_tensors(answers_scores, expected_answers_scores))

    def test_evalai_answer_processor(self):
        evalai_answer_processor = EvalAIAnswerProcessor()

        # Test number
        processed = evalai_answer_processor("two")
        expected = "2"
        self.assertEqual(processed, expected)

        # Test article
        processed = evalai_answer_processor("a building")
        expected = "building"
        self.assertEqual(processed, expected)

        # Test tokenize
        processed = evalai_answer_processor("snow, mountain")
        expected = "snow mountain"
        self.assertEqual(processed, expected)

        # Test contractions
        processed = evalai_answer_processor("isnt")
        expected = "isn't"
        self.assertEqual(processed, expected)

        # Test processor
        processed = evalai_answer_processor("the two mountain's \t \n   ")
        expected = "2 mountain 's"
        self.assertEqual(processed, expected)

    def test_transformer_bbox_processor(self):
        import numpy as np

        config = {
            "params": {
                "bbox_key": "bbox",
                "image_width_key": "image_width",
                "image_height_key": "image_height",
            }
        }

        bbox_processor = TransformerBboxProcessor(config)
        item = {
            "bbox": np.array([[100, 100, 100, 100]]),
            "image_width": 100,
            "image_height": 100,
        }
        processed_box = bbox_processor(item)["bbox"]
        self.assertTrue(
            torch.equal(
                processed_box, torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.float)
            )
        )

    def test_multi_class_from_file(self):
        f = tempfile.NamedTemporaryFile(mode="w", delete=False)
        f.writelines("\n".join(["abc", "bcd", "def", "efg"]))
        f.close()
        config = OmegaConf.create({"vocab_file": f.name})
        processor = MultiClassFromFile(config)

        output = processor({"label": "abc"})
        self.assertEqual(output["class_index"], 0)
        output = processor({"label": "efg"})
        self.assertEqual(output["class_index"], 3)
        output = processor("def")
        self.assertEqual(output["class_index"], 2)

        self.assertRaises(AssertionError, processor, {"label": "UNK"})
        os.unlink(f.name)
