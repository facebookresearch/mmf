# KRISP: Integrating Implicit and Symbolic Knowledge for Open-Domain Knowledge-Based VQA

This repository contains the code for KRISP model from the following paper, released under the MMF:

The majority of KRISP is licensed under CC-BY-NC, however portions of the project are available under separate license terms: TaskGrasp and pytorch\_geometric are licensed under the MIT license.

* K. Marino, X. Chen, D. Parikh, A. Gupta, M. Rohrbach, *KRISP: Integrating Implicit and Symbolic Knowledge for Open-Domain Knowledge-Based VQA*. arXiv preprint, 2020 ([PDF](https://arxiv.org/pdf/2012.11014))
```
@articles{marino2020krisp,
  title={KRISP: Integrating Implicit and Symbolic Knowledge for Open-Domain Knowledge-Based VQA},
  author={Marino, Kenneth and Chen, Xinlei and Parikh, Devi and Gupta, Abhinav and Rohrbach, Marcus},
  booktitle={arXiv preprint arXiv:2012.11014},
  year={2020}
}
```

## Installation
Install the dependencies in requirements.txt or install them manually:
pip install networkx torch_geometric gensim

Check that cuda versions match between your pytorch installation and torch_geometric.

## Training
To train the KRISP model on the OKVQAv1.1 dataset, run the following command
```
mmf_run config=code/mmf_fork_pr/mmf/projects/krisp/configs/krisp/okvqa/train_val.yaml run_type=train_val dataset=okvqa model=krisp
```

For the classic KRISP, we expect to get a OKVQA test accuracy of about 32.31.
There is non-trivial variance between runs, so run at least 3 trials for consistant results.

## Setup / Data
To make sure all data can be found, first run
> export MMF_DATA_DIR=~/.cache/torch/mmf/data
