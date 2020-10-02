# Hateful Memes Dataset

This folder contains configs required to reproduce results and baselines in the [Hateful Memes challenge paper](https://arxiv.org/abs/2005.04790). For details on participation in the challenge, please look at our [Hateful Memes Challenge tutorial](https://mmf.sh/docs/challenges/hateful_memes_challenge).

Please cite the following paper if you use these models and the hateful memes dataset:

* Kiela, D., Firooz, H., Mohan A., Goswami, V., Singh, A., Ringshia P. & Testuggine, D. (2020). *The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes*. arXiv preprint arXiv:2005.04790

```
@article{kiela2020hateful,
  title={The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  author={Kiela, Douwe and Firooz, Hamed and Mohan, Aravind and Goswami, Vedanuj and Singh, Amanpreet and Ringshia, Pratik and  Testuggine, Davide},
  journal={arXiv preprint arXiv:2005.04790},
  year={2020}
}
```
* [Citation for MMF](https://github.com/facebookresearch/mmf/tree/master/README.md#citation)

Links: [[arxiv]](https://arxiv.org/abs/2005.04790) [[challenge]](https://www.drivendata.org/competitions/70/hateful-memes-phase-2/) [[blog post]](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set)

## Prerequisites

Install MMF following the [installation docs](https://mmf.sh/docs/getting_started/installation/).


To acquire the data, you will need to register at DrivenData's Hateful Memes Competition and download data from the challenge's [download page](https://www.drivendata.org/competitions/70/hateful-memes-phase-2/data/). MMF won't be able to automatically download the data since you manually need to agree to the licensing terms. Follow the steps below to convert data into MMF format.

1. Download the zip file with data from DrivenData at location `x`.
2. Note the password from the data download page at DrivenData as `y`
3. Run `mmf_convert_hm --zip_file=x --password=y` where you replace `x` with location of your zip file and `y` with the password you noted.
4. Previous command will unzip and format your files into MMF format. This command can take time while unzipping.
5. Continue, with steps below.

> NOTE: If the command fails with checksum failed, please use `--bypass_checksum=1` option to try without checksum.

## Reproducing Baselines

In the table, we provide configuration corresponding to each of the baselines in the paper. Please use the following individual subsections below for training/evaluating the baselines, submitting to challenge and other things.

**NOTE**: Some of these configurations which depend on region features will auto-download features first time you run the command. (Features are large in size and usually take some time to download and extract)

| Baseline         | Model Key      | Pretrained Key                                   | Config                                                     |
|------------------|----------------|--------------------------------------------------|------------------------------------------------------------|
| Image-Grid       | unimodal_image | unimodal_image.hateful_memes.images              | projects/hateful_memes/configs/unimodal/image.yaml         |
| Image-Region     | unimodal_image | unimodal_image.hateful_memes.features            | projects/hateful_memes/configs/unimodal/with_features.yaml |
| Text BERT        | unimodal_text  | unimodal_text.hateful_memes.bert                 | projects/hateful_memes/configs/unimodal/bert.yaml          |
| Late Fusion      | late_fusion    | late_fusion.hateful_memes                        | projects/hateful_memes/configs/late_fusion/defaults.yaml   |
| ConcatBERT       | concat_bert    | concat_bert.hateful_memes                        | projects/hateful_memes/configs/concat_bert/defaults.yaml   |
| MMBT-Grid        | mmbt           | mmbt.hateful_memes.images                        | projects/hateful_memes/configs/mmbt/defaults.yaml          |
| MMBT-Region      | mmbt           | mmbt.hateful_memes.features                      | projects/hateful_memes/configs/mmbt/with_features.yaml     |
| ViLBERT          | vilbert        | vilbert.finetuned.hateful_memes.direct           | projects/hateful_memes/configs/vilbert/defaults.yaml       |
| Visual BERT      | visual_bert    | visual_bert.finetuned.hateful_memes.direct       | projects/hateful_memes/configs/visual_bert/direct.yaml     |
| ViLBERT CC       | vilbert        | vilbert.finetuned.hateful_memes.from_cc_original | projects/hateful_memes/configs/vilbert/from_cc.yaml        |
| Visual BERT COCO | visual_bert    | visual_bert.finetuned.hateful_memes.from_coco    | projects/hateful_memes/configs/visual_bert/from_coco.yaml  |


For individual baselines and their proper citation have a look at their project pages: [[Visual BERT]](https://github.com/facebookresearch/mmf/tree/master/projects/visual_bert) [[VilBERT]](https://github.com/facebookresearch/mmf/tree/master/projects/vilbert) [[MMBT]](https://github.com/facebookresearch/mmf/tree/master/projects/mmbt)

## Training

Run the following the command, and in general follow MMF's [configuration](https://mmf.sh/docs/notes/configuration) principles to update any parameter if needed:

```
mmf_run config=<REPLACE_WITH_BASELINE_CONFIG> model=<REPLACE_WITH_MODEL_KEY> dataset=hateful_memes
```

This will save the training outputs to an experiment folder under `./save` directory (if not overriden using `env.save_dir`) while running training. The final best model will be saved as `<REPLACE_WITH_MODEL_KEY>_final.pth` inside the experiment folder.

## Evaluation

For running evaluation on validation set run the following command:
```
mmf_run config=<REPLACE_WITH_BASELINE_CONFIG> model=<REPLACE_WITH_MODEL_KEY> dataset=hateful_memes \
run_type=val checkpoint.resume_file=<path_to_best_trained_model> checkpoint.resume_pretrained=False
```

`checkpoint.resume_file` should point to the location of your trained model. This will load the trained model and generate scores on the validation set.

## Predictions for Challenge

For running inference (will generate a csv) on validation set run the following command:

```
mmf_predict config=<REPLACE_WITH_BASELINE_CONFIG> model=<REPLACE_WITH_MODEL_KEY> dataset=hateful_memes \
run_type=val checkpoint.resume_file=<path_to_best_trained_model> checkpoint.resume_pretrained=False
```

For running inference (will generate a csv) on test set run the following command:

```
mmf_predict config=<REPLACE_WITH_BASELINE_CONFIG> model=<REPLACE_WITH_MODEL_KEY> dataset=hateful_memes \
run_type=test checkpoint.resume_file=<path_to_best_trained_model> checkpoint.resume_pretrained=False
```

**NOTE**: All `mmf_predict` commands for the Hateful Memes challenge will generate a csv which you can then submit to DrivenData.

### Predicting for Phase 1

If you want to submit prediction for phase 1, you will need to use command line opts to override the jsonl files that are loaded. At the end of any command you run, add the following to load seen dev and test splits:

```sh
dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_seen.jsonl \
dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_seen.jsonl
```

This will load the phase 1 files for you and evaluate those.

## Evaluating Pretrained Models

For evaluating pretrained models on validation set, use the following command:

```
mmf_run config=<REPLACE_WITH_BASELINE_CONFIG> model=<REPLACE_WITH_MODEL_KEY> dataset=hateful_memes \
run_type=val checkpoint.resume_zoo=<REPLACE_WITH_PRETRAINED_ZOO_KEY> checkpoint.resume_pretrained=False
```

To generate predictions using pretrained models, use the following command:


```
mmf_predict config=<REPLACE_WITH_BASELINE_CONFIG> model=<REPLACE_WITH_MODEL_KEY> dataset=hateful_memes \
run_type=test checkpoint.resume_zoo=<REPLACE_WITH_PRETRAINED_ZOO_KEY> checkpoint.resume_pretrained=False
```

## Loading pretrained model in your code

It is also possible to load a pretrained model directly in your code, use:

```
from mmf.common.registry import registry

model_cls = registry.get_model_class("<DESIRED_MODEL_KEY">)
model = model_cls.from_pretrained("<DESIRED_PRETRAINED_ZOO_KEY>")
```

## Questions/Feedback?

Please open up an [issue on MMF](https://github.com/facebookresearch/mmf/issues/new/choose).
