---
id: concepts
title: Terminology and Concepts
sidebar_label: Terminology and Concepts
---

## Weights and Biases Logger 

MMF has a `WandbLogger` class which lets the user to log their model's progress using [Weights and Biases](https://gitbook-docs.wandb.ai/).

The following options are available in config to enable and customize the wandb logging:
```yaml
training:
    # Weights and Biases control, by default Weights and Biases (wandb) is disabled
    wandb: 
        # Whether to use Weights and Biases Logger, (Default: false)
        enabled: false
        # Project name to be used while logging the experiment with wandb
        wandb_projectname: mmf_${oc.env:USER}
        # Experiment/ run name to be used while logging the experiment 
        # under the project with wandb
        wandb_runname: ${training.experiment_name}
env:
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```
To enable wand logger the user needs to change the following option in the config. 

`training.wandb.enabled=True` 

To give the current experiment a project and run name, user should add these config options.

`training.wandb.wandb_projectname=<ProjectName> training.wandb.wandb_runname=<RunName>`

To change the path to the directory where wandb metadata would be stored (Default: `env.log_dir`):

`env.wandb_logdir=<dir_name>`