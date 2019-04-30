```eval_rst
:github_url: https://github.com/facebookresearch/pythia
```

# Challenge Participation

Participating in EvalAI challenges is really easy using Pythia. We will show how to
do inference for two challenges here:

```eval_rst
.. note::

  This section assumes that you have downloaded data following the Quickstart_ tutorial.

.. _Quickstart: ./quickstart
```
## TextVQA challenge

TextVQA challenge is available at [this link](https://evalai.cloudcv.org/web/challenges/challenge-page/244/overview).
Currently, LoRRA is the SoTA on TextVQA. To do inference on val set using LoRRA, follow the steps below:

```
# Download the model first
cd ~/pythia/data
mkdir -p models && cd models;
# Get link from the table above and extract if needed
wget https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/lorra_best.pth

cd ../..
# Replace tasks, datasets and model with corresponding key for other pretrained models
python tools/run.py --tasks vqa --datasets textvqa --model lorra --config configs/vqa/textvqa/lorra.yml \
--run_type val --evalai_inference 1 --resume_file data/models/lorra_best.pth
```

In the printed log, Pythia will mention where it wrote the JSON file it created.
Upload that file on EvalAI:
```
> Go to https://evalai.cloudcv.org/web/challenges/challenge-page/244/overview
> Select Submit Tab
> Select Validation Phase
> Select the file by click Upload file
> Write a model name
> Upload
```

To check your results, go in 'My submissions' section and select 'Validation Phase' and click on 'Result file'.

Now, you can either edit the LoRRA model to create your own model on top of it or create your own model inside Pythia to
beat LoRRA in challenge.


## VQA Challenge

Similar to TextVQA challenge, VQA Challenge is available at [this link](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview). You can either select Pythia as your base model
or LoRRA model (available soon for VQA2) from the table in [pretrained models](pretrained_models) section as a base.

Follow the same steps above, replacing `--model` with `pythia` or `lorra` and `--datasets` with `vqa2`.
Also, replace the config accordingly. Here are example commands for using Pythia to do inference on test set of VQA2.

```
# Download the model first
cd ~/pythia/data
mkdir -p models && cd models;
# Get link from the table above and extract if needed
wget https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/pythia_train_val.pth

cd ../..
# Replace tasks, datasets and model with corresponding key for other pretrained models
python tools/run.py --tasks vqa --datasets vqa2 --model pythia --config configs/vqa/vqa2/pythia.yml \
--run_type inference --evalai_inference 1 --resume_file data/models/pythia_train_val.pth
```

Now, similar to TextVQA challenge follow the steps to upload the prediction file, but this time to `test-dev` phase.
