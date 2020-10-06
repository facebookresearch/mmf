---
id: concat_bert_tutorial
title: 'Tutorial: Adding a model - Concat BERT'
sidebar_label: Adding a model - Concat BERT
---

In this tutorial, we will go through the step-by-step process of creating a new model using MMF. In this case, we will create a fusion model and train it on the [Hateful Memes dataset](https://github.com/facebookresearch/mmf/tree/master/projects/hateful_memes).

The fusion model that we will create concatenates embeddings from a text encoder and an image encoder and passes them through a two-layer classifier. MMF provides standard image and text encoders out of the box. For the image encoder, we will use ResNet152 image encoder and for the text encoder, we will use BERT-Base Encoder.

## Prerequisites

Follow the prerequisites for installation and dataset [here](https://github.com/facebookresearch/mmf/tree/master/projects/hateful_memes#prerequisites).

## Using MMF to build the model

We will start building our model `ConcatBERTTutorial` using the various building blocks available in MMF. Helper builder methods like `build_image_encoder` for building image encoders, `build_text_encoder` for building text encoders, `build_classifier_layer` for classifier layers etc take configurable params which are defined in the config we will create in the next section. Follow the code and read through the comments to understand how the model is implemented.

```python

import torch

# registry is need to register our new model so as to be MMF discoverable
from mmf.common.registry import registry

# All model using MMF need to inherit BaseModel
from mmf.models.base_model import BaseModel

# Builder methods for image encoder and classifier
from mmf.utils.build import (
    build_classifier_layer,
    build_image_encoder,
    build_text_encoder,
)


# Register the model for MMF, "concat_bert_tutorial" key would be used to find the model
@registry.register_model("concat_bert_tutorial")
class ConcatBERTTutorial(BaseModel):
    # All models in MMF get first argument as config which contains all
    # of the information you stored in this model's config (hyperparameters)
    def __init__(self, config):
        # This is not needed in most cases as it just calling parent's init
        # with same parameters. But to explain how config is initialized we
        # have kept this
        super().__init__(config)
        self.build()

    # This classmethod tells MMF where to look for default config of this model
    @classmethod
    def config_path(cls):
        # Relative to user dir root
        return "configs/models/concat_bert_tutorial/defaults.yaml"

    # Each method need to define a build method where the model's modules
    # are actually build and assigned to the model
    def build(self):
        """
        Config's image_encoder attribute will be used to build an MMF image
        encoder. This config in yaml will look like:

        # "type" parameter specifies the type of encoder we are using here.
        # In this particular case, we are using resnet152
        type: resnet152
        # Parameters are passed to underlying encoder class by
        # build_image_encoder
        params:
            # Specifies whether to use a pretrained version
            pretrained: true
            # Pooling type, use max to use AdaptiveMaxPool2D
            pool_type: avg
            # Number of output features from the encoder, -1 for original
            # otherwise, supports between 1 to 9
            num_output_features: 1
        """
        self.vision_module = build_image_encoder(self.config.image_encoder)

        """
        For text encoder, configuration would look like:
        # Specifies the type of the langauge encoder, in this case mlp
        type: transformer
        # Parameter to the encoder are passed through build_text_encoder
        params:
            # BERT model type
            bert_model_name: bert-base-uncased
            hidden_size: 768
            # Number of BERT layers
            num_hidden_layers: 12
            # Number of attention heads in the BERT layers
            num_attention_heads: 12
        """
        self.language_module = build_text_encoder(self.config.text_encoder)

        """
        For classifer, configuration would look like:
        # Specifies the type of the classifier, in this case mlp
        type: mlp
        # Parameter to the classifier passed through build_classifier_layer
        params:
            # Dimension of the tensor coming into the classifier
            # Visual feature dim + Language feature dim : 2048 + 768
            in_dim: 2816
            # Dimension of the tensor going out of the classifier
            out_dim: 2
            # Number of MLP layers in the classifier
            num_layers: 2
        """
        self.classifier = build_classifier_layer(self.config.classifier)

    # Each model in MMF gets a dict called sample_list which contains
    # all of the necessary information returned from the image
    def forward(self, sample_list):
        # Text input features will be in "input_ids" key
        text = sample_list["input_ids"]
        # Similarly, image input will be in "image" key
        image = sample_list["image"]

        # Get the text and image features from the encoders
        text_features = self.language_module(text)[1]
        image_features = self.vision_module(image)

        # Flatten the embeddings before concatenation
        image_features = torch.flatten(image_features, start_dim=1)
        text_features = torch.flatten(text_features, start_dim=1)

        # Concatenate the features returned from two modality encoders
        combined = torch.cat([text_features, image_features], dim=1)

        # Pass final tensor to classifier to get scores
        logits = self.classifier(combined)

        # For loss calculations (automatically done by MMF
        # as per the loss defined in the config),
        # we need to return a dict with "scores" key as logits
        output = {"scores": logits}

        # MMF will automatically calculate loss
        return output

```

The model’s forward method takes a `SampleList` and outputs a dict containing the logit scores predicted by the model. Different losses and metrics can be calculated on the scores output.

We will define two configs needed for our experiments: (i) a model config for the model's default configurations, and (ii) an experiment config for our particular experiment. The model config provides the model’s default hyperparameters and the experiment config defines and overrides the defaults needed for our particular experiment such as optimizer type, learning rate, maximum updates and batch size.

## Model Config

We will now create the model config file with the params we used while creating the model and store the config in `configs/models/concat_bert_tutorial/defaults.yaml`.

```yaml
model_config:
  concat_bert_tutorial:
    # Type of bert model
    bert_model_name: bert-base-uncased
    direct_features_input: false
    # Dimension of the embedding finally returned by the modal encoder
    modal_hidden_size: 2048
    # Dimension of the embedding finally returned by the text encoder
    text_hidden_size: 768
    # Used when classification head is activated
    num_labels: 2
    # Number of features extracted out per image
    num_features: 1

    image_encoder:
      type: resnet152
      params:
        pretrained: true
        pool_type: avg
        num_output_features: 1

    text_encoder:
      type: transformer
      params:
        bert_model_name: bert-base-uncased
        hidden_size: 768
        num_hidden_layers: 12
        num_attention_heads: 12
        output_attentions: false
        output_hidden_states: false

    classifier:
      type: mlp
      params:
        # 2048 + 768 in case of features
        # Modal_Dim * Number of embeddings + Text Dim
        in_dim: 2816
        out_dim: 2
        hidden_dim: 768
        num_layers: 2
```

## Experiment Config

In the next step, we will create the experiment config which will tell MMF which dataset, optimizer, scheduler, metrics for evalauation to use. We will save this config in `configs/experiments/concat_bert_tutorial/defaults.yaml`:

```yaml
includes:
  - configs/datasets/hateful_memes/bert.yaml

model_config:
  concat_bert_tutorial:
    classifier:
      type: mlp
      params:
        num_layers: 2
    losses:
      - type: cross_entropy

scheduler:
  type: warmup_linear
  params:
    num_warmup_steps: 2000
    num_training_steps: ${training.max_updates}

optimizer:
  type: adam_w
  params:
    lr: 5e-5
    eps: 1e-8

evaluation:
  metrics:
    - accuracy
    - binary_f1
    - roc_auc

training:
  batch_size: 64
  lr_scheduler: true
  max_updates: 22000
  early_stop:
    criteria: hateful_memes/roc_auc
    minimize: false
```

We include the `bert.yaml` config in this as we want to use BERT tokenizer for preprocessing our language data. With both the configs ready we are ready to launch training and evaluation using our model on the Hateful Memes dataset. You can read more about the MMF’s configuration system [here](https://mmf.sh/docs/notes/configuration).

## Training and Evaluation

Now we are ready to train and evaluate our model with the experiment config we created in previous step.

```bash
mmf_run config="configs/experiments/concat_bert_tutorial/defaults.yaml" \
    model=concat_bert_tutorial \
    dataset=hateful_memes \
    run_type=train_val
```

When training ends it will save a final model `concat_bert_tutorial_final.pth` in the experiment folder under `./save` directory. More details about checkpoints can be found [here](https://mmf.sh/docs/tutorials/checkpointing). The command will also generate validation scores after the training gets over.
