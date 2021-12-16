---
id: pytorchvideo
title: Using Pytorchvideo
sidebar_label: Using Pytorchvideo
---

MMF is integrating with Pytorchvideo!

This means you'll be able to use Pytorchvideo models, datasets, and transforms in multimodal models from MMF.
Pytorch datasets and transforms are coming soon!

If you find PyTorchVideo useful in your work, please use the following BibTeX entry for citation.
```
@inproceedings{fan2021pytorchvideo,
    author =       {Haoqi Fan and Tullie Murrell and Heng Wang and Kalyan Vasudev Alwala and Yanghao Li and Yilei Li and Bo Xiong and Nikhila Ravi and Meng Li and Haichuan Yang and  Jitendra Malik and Ross Girshick and Matt Feiszli and Aaron Adcock and Wan-Yen Lo and Christoph Feichtenhofer},
    title = {{PyTorchVideo}: A Deep Learning Library for Video Understanding},
    booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
    year = {2021},
    note = {\url{https://pytorchvideo.org/}},
}
```

## Setup

In order to use pytorchvideo in MMF you need pytorchvideo installed.
You can install pytorchvideo by running
```
pip install pytorchvideo
```
For detailed instructions consult https://github.com/facebookresearch/pytorchvideo/blob/main/INSTALL.md


## Using Pytorchvideo Models in MMF

Currently Pytorchvideo models are supported as MMF encoders.
To use a Pytorchvideo model as the image encoder for your multimodal model,
use the MMF TorchVideoEncoder or write your own encoder that uses pytorchvideo directly.

The TorchVideoEncoder class is a wrapper around pytorchvideo models.
To instantiate a pytorchvideo model as an encoder you can do,

```python
from mmf.modules import encoders
from omegaconf import OmegaConfg

config = OmegaConf.create(
            {
                "name": "torchvideo",
                "model_name": "slowfast_r50",
                "random_init": True,
                "drop_last_n_layers": -1,
            }
        )
encoder = encoders.TorchVideoEncoder(config)

# some video input
fast = torch.rand((1, 3, 32, 224, 224))
slow = torch.rand((1, 3, 8, 224, 224))
output = encoder([slow, fast])
```

In our config object, we specify that we want to build the `torchvideo` (name) encoder,
that we want to use the pytorchvideo model `slowfast_r50` (model_name),
without pretrained weights (`random_init: True`),
and that we want to remove the last module of the network (the transformer head) (`drop_last_n_layers: -1`) to just get the hidden state.
This part depends on which model you're using and what you need it for.

This encoder is usually configured from yaml through your model_config yaml.


Suppose we want to use MViT as our image encoder and we only want the first hidden state.
As the MViT model in pytorchvideo returns hidden states in format (batch size, feature dim, num features),
we want to pass in MViT custom configs and choose the cls pooler.

```python
from mmf.modules import encoders
from omegaconf import OmegaConfg

config = {
            "name": "pytorchvideo",
            "model_name": "mvit_base_32x3",
            "random_init": True,
            "drop_last_n_layers": 0,
            "pooler_name": "cls",
            "spatial_size": 224,
            "temporal_size": 8,
            "head": None,
            "embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
            "atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
            "pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
            "pool_kv_stride_adaptive": [1, 8, 8],
            "pool_kvq_kernel": [3, 3, 3],
        }
encoder = encoders.PytorchVideoEncoder(OmegaConf.create(config))

# some video input
x = torch.rand((1, 3, 8, 224, 224))
output = encoder(x)
```

Here we use the TorchVideoEncoder class to make our MViT model and pick a pooler.
The configs are passed onto the MViT pytorchvideo model.
