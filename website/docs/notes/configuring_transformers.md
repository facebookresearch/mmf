---
id: configuring_transformers
title: 'Configuring Multimodal Transformers'
sidebar_label: Configuring Multimodal Transformers
---

MMF Transformer('mmf_transformer' or 'mmft') is a generalization of multimodal transformer models like MMBT/VisualBERT etc. It provides a customizable framework that supports the following improved usability features :

- Supports an arbitrary number and type of modalities.
- Allows easy switching between different transformer base models (BERT, RoBERTa, XLMR etc.)
- Supports different backend libraries (Huggingface, PyText, Fairseq)
- Pretraining and Finetuning support

In this note, we will go over each aspect and understand how to configure them.



## Configuring Modality Embeddings

MMFT uses three types of embeddings for each modality : feature embedding (input tokens), position embedding(position tokens), type embedding(segment tokens).  first which takes three types of tokens:

- Input ID tokens (modality features)
- Position ID tokens (position embedidng of the modality features)
- Segment ID tokens (token type , to differentiate between the modalities)


Modality specific feature embeddings are generated either during preprocessing the sample or using different image or text encoders available in MMF. Posiiton embedding can also be provided during preprocessing, or MMFT will generate default position embedings. Type embeddings are optional. When added these can either be explicitly specified in the config or MMFT can generate token embeddings in a sequential manner how the modalities are added in the config.

Here is an example config for adding different modalities :


```yaml

model_config:
  mmf_transformer:
    modalities:
      - type: text
        key: text
        position_dim: 128
        segment_id: 0
        layer_norm_eps: 1e-12
        hidden_dropout_prob: 0.1
      - type: image
        key: image
        embedding_dim: 2048
        position_dim: 128
        segment_id: 1
        layer_norm_eps: 1e-12
        hidden_dropout_prob: 0.1
        encoder:
            type: resnet152
            params:
                pretrained: true
                pool_type: avg
                num_output_features: 49
                in_dim: 2048

```

Here is another example that configures MMFT to train on 3 different modalities (text, ocr text and images):

```yaml

model_config:
  mmf_transformer:
    modalities:
      - type: text
        key: text
        position_dim: 64
        segment_id: 0
        layer_norm_eps: 1e-12
        hidden_dropout_prob: 0.1
      - type: text
        key: ocr
        position_dim: 64
        segment_id: 1
        layer_norm_eps: 1e-12
        hidden_dropout_prob: 0.1
      - type: image
        key: image
        embedding_dim: 2048
        position_dim: 64
        segment_id: 2
        layer_norm_eps: 1e-12
        hidden_dropout_prob: 0.1
```

Text (`text`) will have segment ID as 0, ocr text (`ocr`) will have 1 and image (`image`) will have segment ID 2 in order to differentiate between the modalities.

## Configuring Transformer Backends

MMFT leverages integration of different NLP libraries like HUggingface transformers, FairSeq and PyText. MMFT model's base transformer can be built with models from any of these three different libraries. Here is a configuration that uses huggingface backend with MMFT :

```yaml

model_config:
  mmf_transformer:
    transformer_base: bert-base-uncased
    backend:
      type: huggingface
      freeze: false
      params: {}

```

Similarly, for FairSeq backend the configuration can be specified as:

```yaml
model_config:
  mmf_transformer:
    backend:
      type: fairseq
      freeze: false
      model_path: <path_to_fairseq_model>
      params:
        max_seq_len: 254
        num_segments: 1
        ffn_embedding_dim: 3072
        encoder_normalize_before: True
        export: True
        traceable: True
```

:::note

FairSeq and PyText backends are not supported in OSS and will be open sourced in future releases.

:::


## Configuring Transformer Architectures

build_transformer() method is optional to override as base class provides ability load any transformer model from Huggingface transformers just by specifying the name of the model. For example in your model config you can specify

MMFT allows us to change the base transformer architecture easily. When transformer backend is Huggingface, we can choose any transformer model from `transformers` library to build the multimodal model. Here is an example config that specifies the base transformer as Bert Base.

```yaml

model_config:
  mmf_transformer:
   transformer_base: bert-base-uncased

```

Optionally the pretrained weights of this model will be loaded during initialization of the transformer model. Here is another example that uses Roberta as the base transformer :

```yaml

model_config:
  mmf_transformer:
   transformer_base: roberta-base

```

## Configuring Pretraining and Finetuning Heads

[Coming soon]
