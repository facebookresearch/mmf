# Copyright (c) Facebook, Inc. and its affiliates.

import os
import tempfile
from pathlib import Path
from typing import Type, Union

import torch
import torchvision.datasets.folder as tv_helpers
from mmf.common import typings as mmf_typings
from mmf.common.sample import Sample, SampleList
from mmf.models.base_model import BaseModel
from mmf.utils.build import build_processors
from mmf.utils.download import download
from PIL import Image
from torch import nn


MMBT_GRID_HM_CONFIG_PATH = Path("projects/hateful_memes/configs/mmbt/defaults.yaml")
ImageType = Union[Type[Image.Image], str]
PathType = Union[Type[Path], str]
BaseModelType = Type[BaseModel]


class MMBTGridHMInterface(nn.Module):
    """Interface for MMBT Grid for Hateful Memes.
    """

    def __init__(self, model: BaseModelType, config: mmf_typings.DictConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.init_processors()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def init_processors(self):
        config = self.config.dataset_config.hateful_memes
        extra_params = {"data_dir": config.data_dir}
        self.processor_dict = build_processors(config.processors, **extra_params)

    def classify(self, image: ImageType, text: str):
        """Classifies a given image and text in it into Hateful/Non-Hateful.
        Image can be a url or a local path or you can directly pass a PIL.Image.Image
        object. Text needs to be a sentence containing all text in the image.

            >>> from mmf.models.mmbt import MMBT
            >>> model = MMBT.from_pretrained("mmbt.hateful_memes.images")
            >>> model.classify("some_url", "some_text")
            {"label": 0, "confidence": 0.56}

        Args:
            image (ImageType): Image to be classified
            text (str): Text in the image

        Returns:
            bool: Whether image is hateful (1) or non hateful (0)
        """
        if isinstance(image, str):
            if image.startswith("http"):
                temp_file = tempfile.NamedTemporaryFile()
                download(image, *os.path.split(temp_file.name), disable_tqdm=True)
                image = tv_helpers.default_loader(temp_file.name)
                temp_file.close()
            else:
                image = tv_helpers.default_loader(image)

        text = self.processor_dict["text_processor"]({"text": text})
        image = self.processor_dict["image_processor"](image)

        sample = Sample()
        sample.text = text["text"]
        if "input_ids" in text:
            sample.update(text)

        sample.image = image
        sample_list = SampleList([sample])
        device = next(self.model.parameters()).device
        sample_list = sample_list.to(device)

        output = self.model(sample_list)
        scores = nn.functional.softmax(output["scores"], dim=1)
        confidence, label = torch.max(scores, dim=1)

        return {"label": label.item(), "confidence": confidence.item()}
