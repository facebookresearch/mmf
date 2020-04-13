# Sweep Scripts

This folder contains sweep scripts (copied from FAIRSeq, mostly written by [@myleott](https://github.com/myleott) for SLURM. You can use these to run sweeps on scale including grid search based combinations of hyperparameters. Script takes care of normal bootstrapping that has to be done for launching job on SLURM Workload Manager. Adapt [sweep_visual_bert.py](./sweep_visual_bert.py) to create your own script for your custom use case.

An example command to run a sweep on 2 nodes containing 8 GPUs each:

```sh
python tools/sweeps/sweep_visual_bert.py \ --resume_finished \
--resume_failed \
--checkpoints_dir /home/user/jobs \
-t -1 -g 8 -n 2 --constraint some_constraint \
--comment "My run" \
--partition some_partition \
-p my_job
```

Add `--dry_run` argument to first print out what exactly is going to be run without actually running it.
