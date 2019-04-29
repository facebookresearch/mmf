# Pretrained Models

Performing inference using pretrained models in Pythia is easy. Pickup a pretrained
model from the table below and follow the steps to do inference or generate
predictions for EvalAI evaluation. This section expects that you have already installed the
required data as explained in [quickstart](./quickstart).

```eval_rst
+--------+-----------+-----------------------+---------------------------------------+-----------------------------------------------------------+
| Model  | Model Key | Supported Datasets    | Pretrained Models                     | Notes                                                     |
+--------+-----------+-----------------------+---------------------------------------+-----------------------------------------------------------+
| Pythia | pythia    | vqa2, vizwiz, textvqa | `vqa2 train+val`_, `vqa2 train only`_ |                                                           |
+--------+-----------+-----------------------+---------------------------------------+-----------------------------------------------------------+
| LoRRA  | lorra     | vqa2, vizwiz, textvqa | `textvqa`_                            |                                                           |
+--------+-----------+-----------------------+---------------------------------------+-----------------------------------------------------------+
| BAN    | ban       | vqa2, vizwiz, textvqa | Coming soon!                          | Support is preliminary and haven't been tested throughly. |
+--------+-----------+-----------------------+---------------------------------------+-----------------------------------------------------------+

.. _vqa2 train+val: https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/pythia_train_val.pth
.. _vqa2 train only: https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia.pth
.. _textvqa: https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/lorra_best.pth
```

Now, let's say your link to pretrained model `model` is `link` (select from table > right click > copy link address), the respective config should be at
`configs/[task]/[dataset]/[model].yml`. For example, config file for `vqa2 train_and_val` should be
`configs/vqa/vqa2/pythia_train_and_val.yml`. Now to run inference for EvalAI, run the following command.

```
cd ~/pythia/data
mkdir -p models && cd models;
# Download the pretrained model
wget [link]
cd ../..;
python tools/run.py --tasks [task] --datasets [dataset] --model [model] --config [config] \
--run_type inference --evalai_inference 1 --resume_file data/[model].pth
```

If you want to train or evaluate on val, change the `run_type` to `train` or `val`
accordingly. You can also use multiple run types, for e.g. to do training, inference on
val as well as test you can set `--run_type` to `train+val+inference`.

if you remove `--evalai_inference` argument, Pythia will perform inference and provide results
directly on the dataset. Do note that this is not possible in case of test sets as we
don't have answers/targets for them. So, this can be useful for performing inference
on val set locally.
