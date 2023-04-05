# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

from mmf.modules.hf_layers import replace_with_jit, undo_replace_with_jit

try:
    from transformers3.modeling_bert import BertSelfAttention
except ImportError:
    from transformers.modeling_bert import BertSelfAttention


class TestHFLayers(unittest.TestCase):
    def test_undo_replace_with_jit(self):
        original_function = BertSelfAttention.forward
        replace_with_jit()
        undo_replace_with_jit()
        self.assertTrue(BertSelfAttention.forward is original_function)
