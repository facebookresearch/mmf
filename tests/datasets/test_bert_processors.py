# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import tests.test_utils as test_utils
import torch
from omegaconf import OmegaConf


class TestBERTProcessors(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {
                "tokenizer_config": {
                    "type": "bert-base-uncased",
                    "params": {"do_lower_case": True},
                },
                "mask_probability": 0,
                "max_seq_length": 128,
            }
        )

    def test_bert_tokenizer(self):
        from mmf.datasets.processors.bert_processors import BertTokenizer

        test_utils.setup_proxy()
        processor = BertTokenizer(self.config)

        # Test normal caption
        arg = {"text": "This will be a test of tokens?"}
        results = processor(arg)
        expected_input_ids = torch.zeros(128, dtype=torch.long)
        expected_input_ids[:11] = torch.tensor(
            [101, 2023, 2097, 2022, 1037, 3231, 1997, 19204, 2015, 1029, 102],
            dtype=torch.long,
        )
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.zeros(128, dtype=torch.long)
        expected_masks[:11] = 1
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))

        # Test empty caption
        arg = {"text": ""}
        results = processor(arg)
        expected_input_ids = torch.zeros(128, dtype=torch.long)
        expected_input_ids[:2] = torch.tensor([101, 102], dtype=torch.long)
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.zeros(128, dtype=torch.long)
        expected_masks[:2] = 1
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))

        # Test long caption
        arg = {"text": "I am working for facebook " * 100}  # make a long sentence
        results = processor(arg)
        expected_input_ids = [1045, 2572, 2551, 2005, 9130] * 100
        expected_input_ids.insert(0, 101)  # [CLS]
        expected_input_ids = expected_input_ids[:128]
        expected_input_ids[-1] = 102  # [SEP]
        expected_input_ids = torch.tensor(expected_input_ids, dtype=torch.long)
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.ones(128, dtype=torch.long)
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))

        # Test two captions
        arg = {
            "text_a": "This will be a test of tokens?",
            "text_b": "I am working for facebook",
        }
        results = processor(arg)
        expected_input_ids = torch.zeros(128, dtype=torch.long)
        expected_input_ids[:17] = torch.tensor(
            [101, 2023, 2097, 2022, 1037, 3231, 1997, 19204, 2015, 1029, 102]
            + [1045, 2572, 2551, 2005, 9130, 102],
            dtype=torch.long,
        )
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_segment_ids[11:17] = 1
        expected_masks = torch.zeros(128, dtype=torch.long)
        expected_masks[:17] = 1
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))

        # Test masked caption
        processor._probability = 1.0
        arg = {"text": "This will be a test of tokens?"}
        results = processor(arg)
        expected_input_ids = torch.zeros(128, dtype=torch.long)
        expected_input_ids[:11] = torch.tensor(
            [101, 2023, 2097, 2022, 1037, 3231, 1997, 19204, 2015, 1029, 102],
            dtype=torch.long,
        )
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        self.assertFalse(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))

        # Test [MASK] token is present
        self.assertTrue(103 in results["input_ids"])

    def test_vilt_tokenizer(self):
        from mmf.datasets.processors.bert_processors import VILTTextTokenizer

        test_utils.setup_proxy()
        processor = VILTTextTokenizer(self.config)

        # Test normal caption
        arg = {"text": "This will be a test of tokens?"}
        results = processor(arg)
        expected_input_ids = torch.zeros(128, dtype=torch.long)
        expected_input_ids[:11] = torch.tensor(
            [101, 2023, 2097, 2022, 1037, 3231, 1997, 19204, 2015, 1029, 102],
            dtype=torch.long,
        )
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.zeros(128, dtype=torch.long)
        expected_masks[:11] = 1
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))

        # Test empty caption
        arg = {"text": ""}
        results = processor(arg)
        expected_input_ids = torch.zeros(128, dtype=torch.long)
        expected_input_ids[:2] = torch.tensor([101, 102], dtype=torch.long)
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.zeros(128, dtype=torch.long)
        expected_masks[:2] = 1
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))

        # Test long caption
        arg = {"text": "I am working for facebook " * 100}  # make a long sentence
        results = processor(arg)
        expected_input_ids = [1045, 2572, 2551, 2005, 9130] * 100
        expected_input_ids.insert(0, 101)  # [CLS]
        expected_input_ids = expected_input_ids[:128]
        expected_input_ids[-1] = 102  # [SEP]
        expected_input_ids = torch.tensor(expected_input_ids, dtype=torch.long)
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.ones(128, dtype=torch.long)
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))

        # Test two captions
        arg = {
            "text_a": "This will be a test of tokens?",
            "text_b": "I am working for facebook",
        }
        results = processor(arg)
        expected_input_ids = torch.zeros(128, dtype=torch.long)
        expected_input_ids[:17] = torch.tensor(
            [101, 2023, 2097, 2022, 1037, 3231, 1997, 19204, 2015, 1029, 102]
            + [1045, 2572, 2551, 2005, 9130, 102],
            dtype=torch.long,
        )
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_segment_ids[11:17] = 1
        expected_masks = torch.zeros(128, dtype=torch.long)
        expected_masks[:17] = 1
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))

        # Test masked caption
        processor._probability = 1.0
        arg = {"text": "This will be a test of tokens?"}
        results = processor(arg)
        expected_input_ids = torch.zeros(128, dtype=torch.long)
        expected_input_ids[:11] = torch.tensor(
            [101, 2023, 2097, 2022, 1037, 3231, 1997, 19204, 2015, 1029, 102],
            dtype=torch.long,
        )
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        self.assertFalse(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))

        # Test [MASK] token is present
        self.assertTrue(103 in results["input_ids"])

    def test_uniter_tokenizer(self):
        from mmf.datasets.processors.bert_processors import UNITERTextTokenizer

        test_utils.setup_proxy()
        config = OmegaConf.create(
            {
                "tokenizer_config": {
                    "type": "bert-base-uncased",
                    "params": {"do_lower_case": True},
                },
                "mask_probability": 0.5,
                "max_seq_length": 128,
            }
        )

        processor = UNITERTextTokenizer(config)

        # Test normal caption
        arg = {"text": "This will be a test of tokens?"}
        results = processor(arg)
        expected_input_ids = torch.zeros(128, dtype=torch.long)
        expected_input_ids[:11] = torch.tensor(
            [101, 2023, 2097, 2022, 1037, 3231, 1997, 19204, 2015, 1029, 102],
            dtype=torch.long,
        )
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.zeros(128, dtype=torch.long)
        expected_masks[:11] = 1
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))
        self.assertTrue("input_ids_masked" in results)
        self.assertEqual(results["input_ids"].shape, results["input_ids_masked"].shape)

        # Test empty caption
        arg = {"text": ""}
        results = processor(arg)
        expected_input_ids = torch.zeros(128, dtype=torch.long)
        expected_input_ids[:2] = torch.tensor([101, 102], dtype=torch.long)
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.zeros(128, dtype=torch.long)
        expected_masks[:2] = 1
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))
        self.assertTrue("input_ids_masked" in results)
        self.assertEqual(results["input_ids"].shape, results["input_ids_masked"].shape)

        # Test long caption
        arg = {"text": "I am working for facebook " * 100}  # make a long sentence
        results = processor(arg)
        expected_input_ids = [1045, 2572, 2551, 2005, 9130] * 100
        expected_input_ids.insert(0, 101)  # [CLS]
        expected_input_ids = expected_input_ids[:128]
        expected_input_ids[-1] = 102  # [SEP]
        expected_input_ids = torch.tensor(expected_input_ids, dtype=torch.long)
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_masks = torch.ones(128, dtype=torch.long)
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))
        self.assertTrue("input_ids_masked" in results)
        self.assertEqual(results["input_ids"].shape, results["input_ids_masked"].shape)

        # Test two captions
        arg = {
            "text_a": "This will be a test of tokens?",
            "text_b": "I am working for facebook",
        }
        results = processor(arg)
        expected_input_ids = torch.zeros(128, dtype=torch.long)
        expected_input_ids[:17] = torch.tensor(
            [101, 2023, 2097, 2022, 1037, 3231, 1997, 19204, 2015, 1029, 102]
            + [1045, 2572, 2551, 2005, 9130, 102],
            dtype=torch.long,
        )
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        expected_segment_ids[11:17] = 1
        expected_masks = torch.zeros(128, dtype=torch.long)
        expected_masks[:17] = 1
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue(torch.equal(results["input_mask"], expected_masks))
        self.assertTrue("input_ids_masked" in results)
        self.assertEqual(results["input_ids"].shape, results["input_ids_masked"].shape)

        # Test masked caption
        processor._probability = 1.0
        arg = {"text": "This will be a test of tokens?"}
        results = processor(arg)
        expected_input_ids = torch.zeros(128, dtype=torch.long)
        expected_input_ids[:11] = torch.tensor(
            [101, 2023, 2097, 2022, 1037, 3231, 1997, 19204, 2015, 1029, 102],
            dtype=torch.long,
        )
        expected_segment_ids = torch.zeros(128, dtype=torch.long)
        self.assertTrue(torch.equal(results["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(results["segment_ids"], expected_segment_ids))
        self.assertTrue("input_ids_masked" in results)
        self.assertEqual(results["input_ids"].shape, results["input_ids_masked"].shape)

        # Test [MASK] token is present
        self.assertTrue(103 in results["input_ids_masked"])
