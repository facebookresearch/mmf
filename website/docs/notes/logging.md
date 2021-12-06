---
id: logger
title: Weights and Biases Logging
sidebar_label: Weights and Biases Logging
---

## Weights and Biases Logger

MMF now has a `WandbLogger` class which lets the user to log their model's progress using [Weights and Biases](https://wandb.ai/site). Enable this logger to automatically log the training/validation metrics, system (GPU and CPU) metrics and configuration parameters.

## First time setup

To set up wandb, run the following:
```
pip install wandb
```
In order to log anything to the W&B server you need to authenticate the machine with W&B **API key**. You can create a new account by going to https://wandb.ai/signup which will generate an API key. If you are an existing user you can retrieve your key from https://wandb.ai/authorize. You only need to supply your key once, and then it is remembered on the same device.

```
wandb login
```

## W&B config parameters

The following options are available in config to enable and customize the wandb logging:
```yaml
training:
    # Weights and Biases control, by default Weights and Biases (wandb) is disabled
    wandb:
        # Whether to use Weights and Biases Logger, (Default: false)
        enabled: true
        # An entity is a username or team name where you're sending runs.
        # This is necessary if you want to log your metrics to a team account. By default
        # it will log the run to your user account.
        entity: null
        # Project name to be used while logging the experiment with wandb
        project: mmf
        # Experiment/ run name to be used while logging the experiment
        # under the project with wandb
        name: ${training.experiment_name}
        # Specify other argument values that you want to pass to wandb.init(). Check out the documentation
        # at https://docs.wandb.ai/ref/python/init to see what arguments are available.
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
env:
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```

* To enable wandb logger the user needs to change the following option in the config.

    `training.wandb.enabled=True`

* To give the `entity` which is the name of the team or the username, the user needs to change the following option in the config. In case no `entity` is provided, the data will be logged to the `entity` set as default in the user's settings.

    `training.wandb.entity=<teamname/username>`

* To give the current experiment a project and run name, user should add these config options. The default project name is `mmf` and the default run name is `${training.experiment_name}`.

    `training.wandb.project=<ProjectName>` <br />
    `training.wandb.name=<RunName>`

* To change the path to the directory where wandb metadata would be stored (Default: `env.log_dir`):

    `env.wandb_logdir=<dir_name>`

* To provide extra arguments to `wandb.init()`, the user just needs to define them in the config file. Check out the documentation at https://docs.wandb.ai/ref/python/init to see what arguments are available. An example is shown in the config parameter shown above. Make sure to use the same key name in the config file as defined in the documentation.

## Current features

The following features are currently supported by the `WandbLogger`:

* Training & Validation metrics
* Learning Rate over time
* GPU: Type, GPU Utilization, power, temperature, CUDA memory usage
* Log configuration parameters
