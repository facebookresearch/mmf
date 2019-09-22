import unittest
import yaml
import torch
import random
import os
import numpy as np
import pickle

from pythia.models.butd import BUTD
from pythia.common.registry import registry
from pythia.common.sample import Sample, SampleList
from pythia.utils.configuration import ConfigNode, Configuration
from pythia.utils.general import get_pythia_root
from pythia.tasks.processors import VocabProcessor, CaptionProcessor

class TestModelBUTD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        # to test beam search or greedy decoding, simply change config path.
        with open("/test_butd_nucleus_sampling.yaml") as f:
            config = yaml.load(f)

        config = ConfigNode(config)
        # Remove warning
        config.training_parameters.evalai_inference = True
        registry.register("config", config)

        self.config = config

        captioning_config = self.config.task_attributes.captioning.dataset_attributes.coco
        text_processor_config = captioning_config.processors.text_processor
        caption_processor_config = captioning_config.processors.caption_processor

        text_processor_config.params.vocab.vocab_file = "/vocabulary_captioning_thresh5.txt"
        caption_processor_config.params.vocab.vocab_file = "/vocabulary_captioning_thresh5.txt"
        self.text_processor = VocabProcessor(text_processor_config.params)
        self.caption_processor = CaptionProcessor(caption_processor_config.params)

        registry.register("coco_text_processor", self.text_processor)
        registry.register("coco_caption_processor", self.caption_processor)

    def test_forward(self):
        data_dict_path = os.path.join(
            get_pythia_root(), "..", "pythia", "data", "models", "butd.pth"
        )
        state_dict = torch.load(data_dict_path)
        model_config = self.config.model_attributes.butd

        butd = BUTD(model_config)
        butd.build()
        butd.init_losses_and_metrics()

        if list(state_dict.keys())[0].startswith('module') and not hasattr(butd, 'module'):
            state_dict = self._multi_gpu_state_to_single(state_dict)

        butd.load_state_dict(state_dict)
        butd.to("cuda")
        butd.eval()

        self.assertTrue(isinstance(butd, torch.nn.Module))

        test_sample = Sample()
        test_sample.dataset_name = "coco"
        test_sample.dataset_type = "test"

        detectron_features = []

        with open("detectron_features.txt", "rb") as fp:
            detectron_features = pickle.load(fp)

        test_sample.image_feature_0 = detectron_features
        test_sample.answers = torch.zeros((5, 10), dtype=torch.long)


        test_sample_list = SampleList([test_sample])
        test_sample_list = test_sample_list.to("cuda")

        tokens = butd(test_sample_list)["captions"]
        answer = self.caption_processor(tokens.tolist()[0])["caption"]

        # the expected caption is for sum_threshold = 0.5
        expected_caption_nucleus_sampling = "a police officer and a woman on a bike and a motorcycle"

        # to test with other decoding, change to expected_caption_beam_search or expected_caption_greedy
        self.assertMultiLineEqual(answer, expected_caption_nucleus_sampling)

    def _multi_gpu_state_to_single(self, state_dict):
        new_sd = {}
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                raise TypeError("Not a multiple GPU state of dict")
            k1 = k[7:]
            new_sd[k1] = v
        return new_sd

