# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.datasets.processors.bert_processors import MaskedTokenProcessor
from mmf.datasets.processors.frcnn_processor import FRCNNPreprocess
from mmf.datasets.processors.image_processors import TorchvisionTransforms
from mmf.datasets.processors.prediction_processors import ArgMaxPredictionProcessor
from mmf.datasets.processors.processors import (
    BaseProcessor,
    BBoxProcessor,
    CaptionProcessor,
    FastTextProcessor,
    GloVeProcessor,
    GraphVQAAnswerProcessor,
    MultiHotAnswerFromVocabProcessor,
    Processor,
    SimpleSentenceProcessor,
    SimpleWordProcessor,
    SoftCopyAnswerProcessor,
    VocabProcessor,
    VQAAnswerProcessor,
)


__all__ = [
    "BaseProcessor",
    "Processor",
    "VocabProcessor",
    "GloVeProcessor",
    "FastTextProcessor",
    "VQAAnswerProcessor",
    "GraphVQAAnswerProcessor",
    "MultiHotAnswerFromVocabProcessor",
    "SoftCopyAnswerProcessor",
    "SimpleWordProcessor",
    "SimpleSentenceProcessor",
    "BBoxProcessor",
    "CaptionProcessor",
    "MaskedTokenProcessor",
    "TorchvisionTransforms",
    "FRCNNPreprocess",
    "ArgMaxPredictionProcessor",
]
