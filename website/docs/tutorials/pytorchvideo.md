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
                "cls_layer_num": 1,
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
and that we want to remove the last module of the network (the transformer head) (`cls_layer_num: 1`) to just get the hidden state.
This part depends on which model you're using and what you need it for.

This encoder is usually configured from yaml through your model_config yaml.


Suppose we want to use MViT as our image encoder and we only want the first hidden state.
As the MViT model in pytorchvideo returns hidden states in format (batch size, feature dim, num features),
we want to permute the tensor and take the first feature.
To do this we can write our own encoder class in encoders.py

```python
@registry.register_encoder("mvit")
class MViTEncoder(Encoder):
    """
    MVIT from pytorchvideo
    """
    @dataclass
    class Config(Encoder.Config):
        name: str = "mvit"
        random_init: bool = False
        model_name: str = "multiscale_vision_transformers"
        spatial_size: int = 224
        temporal_size: int = 8
        head: Optional[Any] = None

    def __init__(self, config: Config):
        super().__init__()
        self.encoder = TorchVideoEncoder(config)

    def forward(self, *args, **kwargs):
        output = self.encoder(*args, **kwargs)
        output = output.permute(0, 2, 1)
        return output[:, :1, :]
```

Here we use the TorchVideoEncoder class to make our MViT model and transform the output to match what we need from an encoder.
