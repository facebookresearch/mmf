---
id: slurm
title: Large Scale Hyperparameter Sweeps on Slurm
sidebar_label: Sweeping on Slurm
---

MMF provides a utility script for running large scale hyperparameter sweeps on SLURM based cluster setups. A grid search is run on all permutations for the values provided for each of the hyperparameters in the script. The dotlist overrides provided via MMF's configuration system allows to easily override any configuration parameter through this script. This script is created based on sweep scripts provided in FAIRSeq authored by [@myleott](https://github.com/myleott).

An example script to sweep over learning rate and batch size for MMBT on hateful memes would look like (assuming it is living at `tools/sweeps/sweep_mmbt_hm.py`):

```py
import lib as sweep

from lib import hyperparam

def get_grid(args):
    # For list of args, run `python tools/sweeps/sweep_mmbt_hm.py --help`.

    return [
        # Single values (instead of list) remain constant
        hyperparam("model", "mmbt"),
        hyperparam("dataset", "hateful_memes"),
        hyperparam("config", "projects/mmbt/configs/hateful_memes/defaults.yaml"),
        # Grid sweep for 512 and 256
        # save_dir_key value is appended to the folder path for easy recognition
        hyperparam(
            "training.batch_size", [512, 256], save_dir_key=lambda val: f"bs{val}"
        ),
        hyperparam(
            "optimizer.params.lr", [5e-5, 1e-5], save_dir_key=lambda val: f"lr{val}"
        ),
    ]
    # In the above case, sweep will be run over four combinations of batch
    # size and lrs, and folder name would mmbt.bsX.lrY.ngpuZ where
    # number of gpus will be specified via command line argument to this script

def postprocess_hyperparams(args, config):
    # Here you can post process a final config that is generated
    # to adapt some things
    pass


# When directly called through command line, run sweep's main function with
# provided grid configuration
if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
```

An example command to run this sweep on 2 nodes containing each 8 GPUs each would be:

```sh
python tools/sweeps/sweep_mmbt_hm.py:
--resume_finished \
--resume_failed \
--checkpoints_dir /home/user/jobs \
-t -1 \
-g 8 \
-n 2 \
--constraint some_constraint \
--comment "My run" \
--partition some_partition \
-p my_job

# This will log into folder: /home/user/jobs/my_job.bs512.lr5e-05.ngpu16/
# for one of the sweeps among 4.
# Explanation for each of the parameters:
# --resume_finished: Don't skip finished runs, this won't run already running jobs
# --resume_failed: Resume errored out runs from before
# --checkpoints_dir /home/user/jobs: Directory where log folder will be created
# -t -1: Run all sweeps, if t=1, run first combination only
# -g 8: 8 GPUs per node
# -n 2: 2 Nodes in total
# --constraint some_constraint: Slurm constraint if any
# --comment "My run": Slurm comment
# --partition some_partition: Slurm partition
# -p my_job: Prefix for log folder
```

:::tip

Add `--dry_run` argument to first print out what exactly is going to be run without actually running it.

:::

An actual complex sweep config for visual bert with more options can be found at [./tools/sweep/sweep_visual_bert.py](https://github.com/facebookresearch/mmf/blob/master/tools/sweeps/sweep_visual_bert.py). Command following the above command to run it:

```sh
python tools/sweeps/sweep_visual_bert.py \
--resume_finished \
--resume_failed \
--checkpoints_dir /home/user/jobs \
-t -1 -g 8 -n 2 --constraint some_constraint \
--comment "My run" \
--partition some_partition \
-p my_job
```
