# Fine Grained Hateful Memes Dataset

This folder contains configs required to reproduce results and baselines for [2021 ACL WOAH Shared Task Fine Grained Hateful Memes Classification](https://github.com/facebookresearch/fine_grained_hateful_memes). Details of shared task, including background, description, input and output format, can be found through the link.

## Prerequisites

Install MMF following the [installation docs](https://mmf.sh/docs/getting_started/installation/).

To acquire the hateful memes data, follow th instructions [here](https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes).

The additional fine grained labels can be found [here](https://github.com/facebookresearch/fine_grained_hateful_memes/tree/main/data).


## Reproducing Baselines
We provide the configration fine to reproduce the baseline results we have in the [GitHub repo](https://github.com/facebookresearch/fine_grained_hateful_memes). The instrustions for training and evaluation are the same as [hateful memes](https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes). The output format is different and specified in the [GitHub repo](https://github.com/facebookresearch/fine_grained_hateful_memes).

The baselines are all based on VisualBert with image features.

| Baseline         | Model Key      | Pretrained Key                                   | Config                                                     |
|------------------|----------------|--------------------------------------------------|------------------------------------------------------------|
| Hateful       | visual_bert | visual_bert.finetuned.hateful_memes_fine_grained.hateful              | projects/hateful_memes/fine_grained/configs/visual_bert/defaults.yaml         |
| Attack vectors       | visual_bert | visual_bert.finetuned.hateful_memes_fine_grained.attack_vectors              | projects/hateful_memes/fine_grained/configs/visual_bert/attack_vectors.yaml         |
| Protected groups       | visual_bert | visual_bert.finetuned.hateful_memes_fine_grained.protected_groups              | projects/hateful_memes/fine_grained/configs/visual_bert/protected_groups.yaml         |

## Questions/Feedback?

Please open up an [issue on MMF](https://github.com/facebookresearch/mmf/issues/new/choose).
