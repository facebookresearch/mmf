# Copyright (c) Facebook, Inc. and its affiliates.

# Copied from fairseq. Mostly written by @myleott. Adapted accordingly for mmf
import datetime
import itertools
import os
import random
import shlex
import shutil
import subprocess
from collections import OrderedDict
from glob import glob

from mmf.utils.general import get_mmf_root


def main(get_grid, postprocess_hyperparams, args):
    if args.local:
        args.num_nodes = 1

    # compute all possible hyperparameter configurations
    grid = get_grid(args)
    grid_product = list(itertools.product(*[hp.values for hp in grid]))

    # randomly shuffle configurations
    random.seed(args.seed)
    random.shuffle(grid_product)

    for i, hp_values in enumerate(grid_product):
        config = OrderedDict()
        for hp, value in zip(grid, hp_values):
            config[hp.name] = hp
            config[hp.name].current_value = value

        # postprocess hyperparams
        postprocess_hyperparams(args, config)

        # launch training
        job_id = launch_train(args, config)
        if job_id is not None:
            print(f"Launched {job_id}")

        if args.sequential and not args.local and job_id is not None:
            args.dep = job_id

        if i == args.num_trials - 1:
            break


def copy_all_python_files(source, snapshot_main_dir, code_snapshot_hash):
    """
    Copies following files from source to destination:
        a) all *.py files at direct source location.
        b) all mmf/*.py recursively.
    """
    os.makedirs(snapshot_main_dir, exist_ok=True)
    destination = os.path.join(snapshot_main_dir, code_snapshot_hash)
    assert not os.path.exists(
        destination
    ), f"Code snapshot: {code_snapshot_hash} alredy exists"
    os.makedirs(destination)
    all_pys = (
        glob(os.path.join(source, "mmf/**/*.py"), recursive=True)
        + glob(os.path.join(source, "tools/**/*.py"), recursive=True)
        + glob(os.path.join(source, "*.py"))
    )

    for filepath in all_pys:
        directory, filename = os.path.split(filepath)
        if directory:
            os.makedirs(os.path.join(destination, directory), exist_ok=True)
        shutil.copy2(
            os.path.join(source, filepath), os.path.join(destination, filepath)
        )
    return destination


def launch_train(args, config):
    def dry_run(msg):
        if args.dry_run:
            print(f"| dry-run:  {msg}")
        return args.dry_run

    destination = ""
    if args.snapshot_code:
        # Currently hash is just the current time in ISO format.
        code_snapshot_hash = datetime.datetime.now().isoformat()
        destination = copy_all_python_files(
            ".", "slurm_snapshot_code", code_snapshot_hash
        )

    # compute save_dir
    save_dir_key = ".".join(
        filter(
            lambda save_dir_key: save_dir_key is not None,
            [hp.get_save_dir_key() for hp in config.values()],
        )
    )
    save_dir_key = save_dir_key.replace(",", "_")
    num_total_gpus = args.num_nodes * args.num_gpus
    save_dir = os.path.join(
        args.checkpoints_dir, f"{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}"
    )
    tensorboard_logdir = os.path.join(
        args.tensorboard_logdir, f"{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}"
    )

    # create save directory if it doesn"t exist
    if not os.path.exists(save_dir):
        if not dry_run(f"create directory: {save_dir}"):
            os.makedirs(save_dir)

        # copy baseline model
        checkpoint_last = os.path.join(save_dir, "current.ckpt")
        if (
            args.baseline_model
            and not os.path.exists(checkpoint_last)
            and not dry_run(f"initialize with baseline model: {args.baseline_model}")
        ):
            if not os.path.exists(args.baseline_model):
                raise FileNotFoundError(
                    f"Cannot find baseline model: {args.baseline_model}"
                )
            shutil.copyfile(args.baseline_model, checkpoint_last)

    # check for whether the run failed
    if has_finished(save_dir):
        if args.resume_finished:
            dry_run(f"restart previously finished run: {save_dir}")
        else:
            print(f"skip finished run (override with --resume-finished): {save_dir}")
            return
    elif has_failed(save_dir):
        if args.resume_failed:
            dry_run(f"resume failed run: {save_dir}")
        else:
            print(f"skip failed run (override with --resume-failed): {save_dir}")
            return
    elif has_started(save_dir):
        print(f"skip in progress run: {save_dir}")
        return

    # generate train command
    train_cmd = [
        "python",
        "-u",
        os.path.join(get_mmf_root(), "..", "mmf_cli", "run.py"),
    ]
    train_cmd.extend(["distributed.world_size", str(args.num_nodes * args.num_gpus)])
    if args.num_nodes > 1:
        train_cmd.extend(["distributed.port", str(get_random_port())])

    if args.config is not None:
        train_cmd.extend(["config", args.config])
    train_cmd.extend(["checkpoint.resume", "True"])
    train_cmd.extend(["env.save_dir", save_dir])
    if args.tensorboard:
        train_cmd.extend(["training.tensorboard", "1"])
        train_cmd.extend(["env.tensorboard_logdir", tensorboard_logdir])
    for hp in config.values():
        train_cmd.extend(map(str, hp.get_cli_args()))
    if args.extra_args is not None and len(args.extra_args) > 0:
        # convert commands with equal sign to the other format without the equal sign
        # e.g. ["training.batch_size=128"] to ["training.batch_size", "128"]
        extra_args = [c for arg in args.extra_args for c in arg.split("=")]
        train_cmd.extend(extra_args)
    if args.dry_run:
        print(train_cmd)
        train_cmd_str = " ".join(train_cmd)
        dry_run(f"train command: {train_cmd_str}")

    # start training
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    if args.local:
        assert (
            args.num_nodes == 1
        ), "distributed training cannot be combined with --local"
        if not dry_run("start training locally"):
            if "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args.num_gpus)))
            env["NCCL_DEBUG"] = "INFO"
            train_proc = subprocess.Popen(train_cmd, env=env)
            train_proc.wait()
    else:
        train_log = os.path.join(save_dir, "train.log")
        train_stderr = os.path.join(save_dir, "train.stderr.%j")  # %j = slurm job id

        # set environment
        if args.num_nodes > 1:
            env["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
            env["NCCL_DEBUG"] = "INFO"
        else:
            env["NCCL_SOCKET_IFNAME"] = ""

        srun_cmd = [
            "srun",
            "--job-name",
            f"{args.prefix}.{save_dir_key}",
            "--output",
            train_log,
            "--error",
            train_stderr,
            "--open-mode",
            "append",
            "--unbuffered",
        ]
        if args.salloc:
            srun_cmd += [
                "--nodes",
                str(args.num_nodes),
                "--ntasks",
                str(args.num_nodes),
            ]
        srun_cmd += train_cmd
        srun_cmd_str = " ".join(map(shlex.quote, srun_cmd)) + " &"

        # build command
        if not args.salloc:
            excluded_hosts = os.environ.get("EXCLUDED_HOSTS", None)
            included_hosts = os.environ.get("INCLUDED_HOSTS", None)
            gres = f"gpu:{args.gpu_type}:{args.num_gpus}"
            sbatch_cmd = [
                "sbatch",
                "--job-name",
                f"{args.prefix}.{save_dir_key}",
                "--gres",
                gres,
                "--nodes",
                str(args.num_nodes),
                "--ntasks-per-node",
                "1",
                "--cpus-per-task",
                str(int(8 * args.num_gpus)),
                "--output",
                train_log,
                "--error",
                train_stderr,
                "--open-mode",
                "append",
                # "--no-requeue",
                "--signal",
                "B:USR1@180",
            ]
            if args.constraint:
                sbatch_cmd += ["-C", args.constraint]

            if args.partition:
                sbatch_cmd += ["--partition", args.partition]
            if args.reservation:
                sbatch_cmd += ["--reservation", args.reservation]
            if args.exclusive:
                sbatch_cmd += ["--exclusive"]
            if args.comment:
                comment = args.comment
                if args.snapshot_code:
                    comment += f", Code Location: {destination}"
                sbatch_cmd += ["--comment", comment]

            if args.snapshot_code:
                sbatch_cmd += ["--comment", f"Code Location: {destination}"]

            if args.dep is not None:
                sbatch_cmd.extend(["-d", str(args.dep)])
            if args.time is not None:
                sbatch_cmd.extend(["--time", args.time])
            if args.mem is not None:
                sbatch_cmd += ["--mem", args.mem]
            else:
                sbatch_cmd += ["--mem-per-cpu", "7G"]
            sbatch_cmd += ["-x", excluded_hosts] if excluded_hosts is not None else []
            sbatch_cmd += ["-w", included_hosts] if included_hosts is not None else []

            wrapped_cmd = (
                requeue_support()
                + "\n"
                + srun_cmd_str
                + " \n wait $! \n sleep 610 & \n wait $!"
            )

            sbatch_cmd += ["--wrap", wrapped_cmd]
            sbatch_cmd_str = " ".join(map(shlex.quote, sbatch_cmd))
        else:
            sbatch_cmd = srun_cmd
            sbatch_cmd_str = srun_cmd_str

        if args.dry_run:
            dry_run("start remote training")
            dry_run(f"- log stdout to: {train_log}")
            dry_run(f"- log stderr to: {train_stderr}")
            dry_run(f"- run command: {sbatch_cmd_str}")
            sbatch_cmd += ["--test-only"]
            with subprocess.Popen(
                sbatch_cmd, stdout=subprocess.PIPE, env=env
            ) as train_proc:
                stdout = train_proc.stdout.read().decode("utf-8")
                print(stdout)
        else:
            with open(train_log, "a") as train_log_h:
                # log most recent git commit
                git_commit = subprocess.check_output(
                    "git log | head -n 1", shell=True, encoding="utf-8"
                )
                print(git_commit.rstrip(), file=train_log_h)
                if args.baseline_model:
                    print(f"baseline model: {args.baseline_model}", file=train_log_h)
            with open(train_log, "a") as train_log_h:
                print(f"running command: {sbatch_cmd_str}\n")
                print(f"running command: {sbatch_cmd_str}\n", file=train_log_h)
                with subprocess.Popen(
                    sbatch_cmd, stdout=subprocess.PIPE, env=env
                ) as train_proc:
                    stdout = train_proc.stdout.read().decode("utf-8")
                    print(stdout, file=train_log_h)
                    try:
                        job_id = int(stdout.rstrip().split()[-1])
                        return job_id
                    except IndexError:
                        return None


def has_finished(save_dir):
    train_log = os.path.join(save_dir, "train.log")
    if not os.path.exists(train_log):
        return False
    with open(train_log) as h:
        lines = h.readlines()
        if len(lines) == 0:
            return False
        if "Finished run" in lines[-1]:
            return True
    return False


def has_failed(save_dir):
    if not os.path.exists(save_dir):
        return False

    # find max job id
    job_ids = []
    for fn in os.listdir(save_dir):
        if fn.startswith("train.stderr."):
            job_ids.append(int(fn.split(".")[-1]))
    if len(job_ids) == 0:
        return False
    max_job_id = max(job_ids)

    def _has_failed(stderr_fn):
        with open(stderr_fn) as h:
            for line in h:
                if len(line.strip()) > 0:
                    # assume that any output in stderr indicates an error
                    return True
        return False

    return _has_failed(os.path.join(save_dir, f"train.stderr.{max_job_id}"))


def has_started(save_dir):
    train_log = os.path.join(save_dir, "train.log")
    if not os.path.exists(train_log):
        return False
    return True


def get_random_port():
    old_state = random.getstate()
    random.seed()
    port = random.randint(10000, 20000)
    random.setstate(old_state)
    return port


def requeue_support():
    return """
        trap_handler () {
           echo "Caught signal: " $1
           # SIGTERM must be bypassed
           if [ "$1" = "TERM" ]; then
               echo "bypass sigterm"
           else
             # Submit a new job to the queue
             echo "Requeuing " $SLURM_JOB_ID
             scontrol requeue $SLURM_JOB_ID
           fi
        }


        # Install signal handler
        trap 'trap_handler USR1' USR1
        trap 'trap_handler TERM' TERM
    """
