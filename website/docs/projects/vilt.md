---
id: vilt
sidebar_label: ViLT
title: "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision"
---

This repository contains the code for pytorch implementation of ViLT model, released originally under this ([repo](https://github.com/dandelin/ViLT)). Please cite the following papers if you are using ViLT model from mmf:

* Wonjae Kim, Bokyung Son, and Ildoo Kim. 2021. *ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision}*. In 38th International Conference on Machine Learning (ICML). ([arXiV](https://arxiv.org/pdf/2102.03334))
```
@misc{kim2021vilt,
      title={ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision},
      author={Wonjae Kim and Bokyung Son and Ildoo Kim},
      year={2021},
      eprint={2102.03334},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html).

## Training

To train ViLT model from scratch on the VQA2.0 dataset, run the following command
```
mmf_run config=projects/vilt/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=vilt
```

To finetune using different pretrained starting weights, change the `pretrained_model_name` under image_encoder in the config yaml to reference a huggingface model.

To finetrain a pretrained ViLT model on the VQA2.0 dataset,
```
mmf_run config=projects/vilt/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=vilt checkpoint.resume_zoo=vilt.pretrained
```

To test a ViLT model already finetuned on the VQA2.0 dataset,
```
mmf_run config=projects/vilt/configs/vqa2/defaults.yaml run_type=val dataset=vqa2 model=vilt checkpoint.resume_zoo=vilt.vqa
```

To pretrain a ViLT model from scratch on the COCO dataset,
```
mmf_run config=projects/vilt/configs/masked_coco/pretrain.yaml run_type=train_val dataset=masked_coco model=vilt
```

## Using the ViLT model from code
Here is an example of running the ViLT model from code, to do visual question answering (vqa) on a raw image and text.
The forward pass takes ~15ms which is very fast compared to UNITER's ~600ms.

```python
from argparse import Namespace

import torch
from mmf.common.sample import SampleList
from mmf.datasets.processors.bert_processors import VILTTextTokenizer
from mmf.datasets.processors.image_processors import VILTImageProcessor
from mmf.utils.build import build_model
from mmf.utils.configuration import Configuration, load_yaml
from mmf.utils.general import get_current_device
from mmf.utils.text import VocabDict
from omegaconf import OmegaConf
from PIL import Image
```

A way to make model configs and instantiate the ViLT model.
```python
# make model config for vilt vqa2
model_name = "vilt"
config_args = Namespace(
    config_override=None,
    opts=["model=vilt", "dataset=vqa2", "config=configs/defaults.yaml"],
)
default_config = Configuration(config_args).get_config()
model_vqa_config = load_yaml(
    "/private/home/your/path/to/mmf/projects/vilt/configs/vqa2/defaults.yaml"
)
config = OmegaConf.merge(default_config, model_vqa_config)
OmegaConf.resolve(config)
model_config = config.model_config[model_name]
model_config.model = model_name
vilt_model = build_model(model_config)
```

Load model weights, `model_checkpoint_path` is the model checkpoint downloaded at model zoo path `vilt.vqa`,
with current url `s3://dl.fbaipublicfiles.com/mmf/data/models/vilt/vilt.finetuned.vqa2.tar.gz`
```python
# build model and load weights
model_checkpoint_path = './vilt_vqa2.pth'
state_dict = torch.load(model_checkpoint_path)
vilt_model.load_state_dict(state_dict, strict=False)
vilt_model.eval()
vilt_model = vilt_model.to(get_current_device())
```

Prepare input image and text.
This example is using an image of a man with a hat kissing his daughter.
The text is the question posed to the ViLT model for visual question answering.
```python
# get image input
image_processor = VILTImageProcessor({"size": [384, 384]})
image_path = "./kissing_image.jpg"
raw_img = Image.open(image_path).convert("RGB")
image = image_processor(raw_img)

# get text input
text_tokenizer = VILTTextTokenizer({})
question = "What is on his head?"
processed_text_dict = text_tokenizer({"text": question})
```

Wrap everything up in a sample list as expected by the ViLT BaseModel.
```python
# make batch inputs
sample_dict = {**processed_text_dict, "image": image}
sample_dict = {
    k: v.unsqueeze(0) for k, v in sample_dict.items() if isinstance(v, torch.Tensor)
}
sample_dict["targets"] = torch.zeros((1, 3129))
sample_dict["targets"][0,1358] = 1
sample_dict["dataset_name"] = "vqa2"
sample_dict["dataset_type"] = "test"
sample_list = SampleList(sample_dict).to(get_current_device())
```

Load the vqa answer -> word string map to understand what it says!
Currently file url at `s3://dl.fbaipublicfiles.com/mmf/data/datasets/vqa2/defaults/extras/vocabs/answers_vqa.txt`
```python
# load vqa2 id -> answers
vocab_file_path = "/private/home/path/to/answers_vqa.txt"
answer_vocab = VocabDict(vocab_file_path)
```

And heres the part you've been waiting for!
```python
# do prediction
with torch.no_grad():
    vqa_logits = vilt_model(sample_list)["scores"]
    answer_id = vqa_logits.argmax().item()
    answer = answer_vocab.idx2word(answer_id)
    print(chr(27) + "[2J") # clear the terminal
    print(f"{question}: {answer}")
```

Expected output `What is on his head?: hat`
