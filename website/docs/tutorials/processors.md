---
id: processors
title: 'Adding a Processor'
sidebar_label: Adding a Processor
---

Processors can be thought of as torchvision transforms which transform a sample into a form usable by the model. Each processor takes in a dictionary and returns a dictionary. Processors are initialized as member variables of the dataset and can be used to preprocess samples in the proper format. Here is how processors work in mmf:

<div align="center">
  <img width="80%" src="https://i.imgur.com/9sZTiUp.gif"/>
</div>

For this tutorial, we will create three different types of processors :

1. a simple processor for text,
2. a simple processor for images,
3. a text processor by extending an existing vocabulary processor in mmf,

## Create a simple Text Processor

Here we will create a simple processor that takes a sentence and returns a list of stripped word tokens.

```python

# registry is needed to register the processor so it is discoverable by MMF
from mmf.common.registry import registry
# We will inherit the BaseProcessor in MMF
from mmf.datasets.processors import BaseProcessor

@registry.register_processor("simple_processor")
class SimpleProccessor(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        return

    # Override the call method
    def __call__(self, item):
        text = item['text']
        text = [t.strip() for t in text.split(" ")]
        return {"text": text}
```

We can add the processor's configuration to a dataset's config and will be available in the dataset class as `text_processor` variable:

```yaml
dataset_config:
  vqa2:
    processors:
      text_processor:
        type: simple_processor
```

In this manner, processors can be added to any dataset.

## Create an Image Processor

In this section, we will learn how to add an image processor. We will add a processor that converts any grayscale images to 3 channel image.

```python

import torch

# registry is needed to register the processor so it is discoverable by MMF
from mmf.common.registry import registry
# We will inherit the BaseProcessor in MMF
from mmf.datasets.processors import BaseProcessor

@registry.register_processor("GrayScale")
class GrayScale(BaseProcessor):
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, item):
        return self.transform(item["image"])

    def transform(self, x):
        assert isinstance(x, torch.Tensor)
        # Handle grayscale, tile 3 times
        if x.size(0) == 1:
            x = torch.cat([x] * 3, dim=0)
        return x

```

We will add the processor's configuration to the Hateful Memes dataset's config:

```yaml
dataset_config:
  vqa2:
    processors:
      image_processor:
        type: torchvision_transforms
        params:
          transforms:
            - ToTensor
            - GrayScale
```

The `torchvision_transforms` image processor loads the different transform processor like the `GrayScale` one we created and composes them together as torchvision transforms. Here we are adding two transforms, first `ToTensor`, which is a native torchvision transform to convert the image to a tensor and then the second `GrayScale` which will convert a single channel to 3 channel image tensor. So these transforms will be applied to the images when `image_processor` is used on an image from the dataset class.

## Extending an existing processor: Create a fasttext sentence processor

A [`fasttext`](https://github.com/facebookresearch/mmf/blob/f11adf0e4a5a28e85239176c44342f6471550e84/mmf/datasets/processors/processors.py#L361) processor is available in MMF that returns word embeddings. Here we will create a `fasttext` _sentence_ processor hereby extending the `fasttext` word processor.

```python

import torch

# registry is needed to register the processor so it is discoverable by MMF
from mmf.common.registry import registry
# We will inherit the FastText Processor already present in MMF.
# FastTextProcessor inherits from VocabProecssor
from mmf.datasets.processors import FastTextProcessor


# Register the processor so that MMF can discover it
@registry.register_processor("fasttext_sentence_vector")
class FastTextSentenceVectorProcessor(FastTextProcessor):
   # Override the call method
   def __call__(self, item):
       # This function is present in FastTextProcessor class and loads
       # fasttext bin
       self._load_fasttext_model(self.model_file)
       if "text" in item:
           text = item["text"]
       elif "tokens" in item:
           text = " ".join(item["tokens"])

       # Get a sentence vector for sentence and convert it to torch tensor
       sentence_vector = torch.tensor(
           self.model.get_sentence_vector(text),
           dtype=torch.float
       )
       # Return back a dict
       return {
           "text": sentence_vector
       }

   # Make dataset builder happy, return a random number
   def get_vocab_size(self):
       return None
```

For this processor, we can similarly add the configuration to the a dataset's config and will be available in the dataset class as `text_processor` :

```yaml
dataset_config:
  vqa2:
    processors:
      text_processor:
        type: fasttext_sentence_vector
        params:
          max_length: null
          model_file: wiki.en.bin
```

## Next Steps

Learn more about processors in the [processors documentation](https://mmf.sh/api/lib/datasets/processors.html).
