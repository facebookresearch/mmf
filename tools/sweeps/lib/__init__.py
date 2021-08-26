# Copyright (c) Facebook, Inc. and its affiliates.

# Copied from fairseq. Mostly written by @myleott. Adapted accordingly for mmf
import argparse
import datetime
import os
import socket


# if argv is None, we will read from sys.argv (invoke params)
def get_args(argv=None):
    parser = argparse.ArgumentParser("Script for launching hyperparameter sweeps")
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
    )
    parser.add_argument(
        "-t",
        "--num_trials",
        required=True,
        type=int,
        help="number of random hyperparam configurations to try (-1 for grid search)",
    )
    parser.add_argument(
        "-g", "--num_gpus", type=int, required=True, help="number of GPUs per node"
    )
    parser.add_argument(
        "-n",
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="aicommerce__multimodal_model",
        help="registered model type",
    )
    parser.add_argument(
        "--oncall", type=str, default="ai_commerce", help="oncall team "
    )
    parser.add_argument(
        "--capabilities",
        type=str,
        default="GPU_V100_HOST",
        help="hardware capabilities",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--config", type=str, default=None, help="configuration for model"
    )
    parser.add_argument(
        "--extra_args",
        type=str,
        nargs="*",
        help="extra arguments to be passed into MMF command (e.g. config arguments)",
    )
    parser.add_argument(
        "--baseline_model", help="path to baseline model from which to resume training"
    )
    parser.add_argument(
        "--force_checkpoints_dir", help="force using a given checkpoint dir"
    )
    parser.add_argument(
        "--resume_failed",
        action="store_true",
        help="resume any runs that failed (assumes --num_trials and --seed"
        + " are the same)",
    )
    parser.add_argument(
        "--resume_finished",
        action="store_true",
        help="force any runs that finished to begin again (uncommon)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="output only a list of actions to perform without performing them",
    )
    parser.add_argument("--local", action="store_true", help="run job locally")
    parser.add_argument("--debug", action="store_true", help="debug")

    hostname = socket.gethostname()
    if "fair" in hostname:
        default_backend = "slurm"
        parser.add_argument(
            "--checkpoints_dir",
            default=os.path.join(
                "/checkpoint", os.environ["USER"], str(datetime.date.today())
            ),
            help="save checkpoints and logs in "
            + "<checkpoints-dir>/<prefix>.<save_dir_key>",
        )
    else:
        default_backend = "fblearner"
        parser.add_argument(
            "--checkpoints_dir",
            default=os.path.join(
                "/mnt/vol/gfsai-east/ai-group/users",
                os.environ["USER"],
                "checkpoints",
                str(datetime.date.today()),
            ),
            help="save checkpoints and logs in "
            + "<checkpoints-dir>/<prefix>.<save_dir_key>",
        )
        parser.add_argument(
            "--workflow",
            default="faim.mmf_run.train_workflow@faim",
            help="fblearner workflow name",
        )
        parser.add_argument(
            "--buck-target", default=None, help="fblearner buck-target if required"
        )

    parser.add_argument(
        "--backend", choices=["slurm", "fblearner"], default=default_backend
    )

    # FBLearner params
    parser.add_argument(
        "--entitlement", help="entitlement to use", default="bigbasin_atn_fair"
    )
    parser.add_argument(
        "--run-as-secure-group",
        help="secure group to use",
        default="fair_research_and_engineering",
    )

    # Slurm params
    parser.add_argument(
        "--salloc", action="store_true", help="run agaist current allocation"
    )
    parser.add_argument("--partition", help="partition to run on", default="learnfair")
    parser.add_argument("--reservation", help="reservation to run on")
    parser.add_argument(
        "--exclusive", action="store_true", help="if set, get exclusive host"
    )
    parser.add_argument(
        "--dep",
        metavar="JOBID",
        type=int,
        help="add JOBID as a dependency (i.e., wait for it to finish)",
    )
    parser.add_argument(
        "--sequential", action="store_true", help="schedule jobs to run sequentially"
    )
    parser.add_argument(
        "--time", default="4320", help="expected job duration in minutes"
    )
    parser.add_argument("--mem", "--mem", help="memory to request")
    parser.add_argument("--gpu-type", default="volta")
    parser.add_argument(
        "--constraint",
        metavar="CONSTRAINT",
        help="gpu constraint, if any. e.g. 'volta'",
    )
    parser.add_argument("--comment", help="comment string")
    parser.add_argument(
        "--snapshot_code",
        action="store_true",
        default=False,
        help="Flag for creating a snapshot of training code while creating slurm job,"
        " path is './slurm_snapshot_code/<TIME_ISO_FORMAT/>:', "
        "can find time from comment of slurm job.",
    )
    parser.add_argument(
        "--tensorboard_logdir",
        default=os.path.join(
            "/checkpoint",
            os.environ["USER"],
            "tensorboard_logs",
            str(datetime.date.today()),
        ),
        help="save tensorboard logs in <tensorboard-logdir>/<prefix>.<save_dir_key>",
    )
    parser.add_argument(
        "--tensorboard",
        default=0,
        type=int,
        help="enable tensorboard logging by passing --tensorboard 1",
    )

    # Will read sys.argv if argv is None
    args = parser.parse_args(argv)
    return args


class hyperparam:
    """Base class for defining hyperparameters."""

    def __init__(self, name, values=None, binary_flag=False, save_dir_key=None):
        """
        Arguments:
        - name : the name of the hyperparameter (e.g., `--dropout`)
        - values : the set of values to sweep over (e.g., `[0.0, 0.1, 0.2]`)
        - binary_flag : whether the hyperparameter uses a boolean flag
                        (e.g., `--no-save`)
        - save_dir_key : function that takes the hyperparameter value and returns
                        the "key" to be appended to the output directory name
        """
        self.name = name
        if values is None:  # syntactic sugar for binary flags
            self.values = [True]
            self.binary_flag = True
        else:
            self.values = values if isinstance(values, list) else [values]
            self.binary_flag = binary_flag
        self.save_dir_key = save_dir_key
        self.current_value = None

        if len(self.values) > 1 and self.save_dir_key is None:
            raise ValueError(
                f"{name} has more than one value but is missing a save_dir_key!"
            )

    def get_cli_args(self):
        if self.binary_flag:
            return [self.name] if self.current_value else []
        else:
            return [self.name, self.current_value]

    def get_save_dir_key(self):
        if self.save_dir_key is None:
            return None
        if self.binary_flag:
            return self.save_dir_key(1) if self.current_value else None
        return self.save_dir_key(self.current_value)


def main(get_grid, postprocess_hyperparams):
    args = get_args()

    if args.backend == "slurm":
        from .slurm import main as backend_main
    elif args.backend == "fblearner":
        from .fblearner import main as backend_main

    backend_main(get_grid, postprocess_hyperparams, args)
