# Copyright (c) Facebook, Inc. and its affiliates.

import gc
import math
import time

import omegaconf
import torch
from tqdm import tqdm

from mmf.common.dataset_loader import DatasetLoader
from mmf.common.meter import Meter
from mmf.common.registry import registry
from mmf.common.report import Report
from mmf.modules.metrics import Metrics
from mmf.utils.build import build_model, build_optimizer, build_scheduler
from mmf.utils.checkpoint import Checkpoint
from mmf.utils.configuration import get_mmf_env
from mmf.utils.distributed import (
    broadcast_scalar,
    get_world_size,
    is_master,
    reduce_dict,
)
from mmf.utils.early_stopping import EarlyStopping
from mmf.utils.general import clip_gradients, print_model_parameters
from mmf.utils.logger import Logger, TensorboardLogger
from mmf.utils.timer import Timer


@registry.register_trainer("base_trainer")
class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.profiler = Timer()
        self.total_timer = Timer()

    def load(self):
        self._set_device()

        self.run_type = self.config.get("run_type", "train")
        self.dataset_loader = DatasetLoader(self.config)
        self._datasets = self.config.datasets

        # Check if loader is already defined, else init it
        writer = registry.get("writer", no_warning=True)
        if writer:
            self.writer = writer
        else:
            self.writer = Logger(self.config)
            registry.register("writer", self.writer)

        configuration = registry.get("configuration", no_warning=True)
        if configuration:
            configuration.pretty_print()

        self.config_based_setup()

        self.load_datasets()
        self.load_model_and_optimizer()
        self.load_metrics()

    def _set_device(self):
        self.local_rank = self.config.device_id
        self.device = self.local_rank
        self.distributed = False

        # Will be updated later based on distributed setup
        registry.register("global_device", self.device)

        if self.config.distributed.init_method is not None:
            self.distributed = True
            self.device = torch.device("cuda", self.local_rank)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        registry.register("current_device", self.device)

    def load_datasets(self):
        self.writer.write("Loading datasets", "info")
        self.dataset_loader.load_datasets()

        self.train_dataset = self.dataset_loader.train_dataset
        self.val_dataset = self.dataset_loader.val_dataset

        # Total iterations for snapshot
        self.snapshot_iterations = len(self.val_dataset)
        self.snapshot_iterations //= self.config.training.batch_size

        self.test_dataset = self.dataset_loader.test_dataset

        self.train_loader = self.dataset_loader.train_loader
        self.val_loader = self.dataset_loader.val_loader
        self.test_loader = self.dataset_loader.test_loader

    def load_metrics(self):
        metrics = self.config.evaluation.get("metrics", [])
        self.metrics = Metrics(metrics)
        self.metrics_params = self.metrics.required_params

    def load_model_and_optimizer(self):
        attributes = self.config.model_config[self.config.model]
        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_config[attributes]

        with omegaconf.open_dict(attributes):
            attributes.model = self.config.model

        self.model = build_model(attributes)

        if "cuda" in str(self.device):
            device_info = "CUDA Device {} is: {}".format(
                self.config.distributed.rank,
                torch.cuda.get_device_name(self.local_rank),
            )
            registry.register("global_device", self.config.distributed.rank)
            self.writer.write(device_info, log_all=True)

        self.model = self.model.to(self.device)
        self.optimizer = build_optimizer(self.model, self.config)

        registry.register("data_parallel", False)
        registry.register("distributed", False)

        self.load_extras()
        self.parallelize_model()

    def parallelize_model(self):
        training = self.config.training
        if (
            "cuda" in str(self.device)
            and torch.cuda.device_count() > 1
            and not self.distributed
        ):
            registry.register("data_parallel", True)
            self.model = torch.nn.DataParallel(self.model)

        if "cuda" in str(self.device) and self.distributed:
            registry.register("distributed", True)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                check_reduction=True,
                find_unused_parameters=training.find_unused_parameters,
            )

    def load_extras(self):
        self.writer.write("Torch version is: " + torch.__version__)
        self.checkpoint = Checkpoint(self)
        self.meter = Meter()

        self.training_config = self.config.training

        early_stop_criteria = self.training_config.early_stop.criteria
        early_stop_minimize = self.training_config.early_stop.minimize
        early_stop_enabled = self.training_config.early_stop.enabled
        early_stop_patience = self.training_config.early_stop.patience

        self.log_interval = self.training_config.log_interval
        self.evaluation_interval = self.training_config.evaluation_interval
        self.checkpoint_interval = self.training_config.checkpoint_interval
        self.max_updates = self.training_config.max_updates
        self.should_clip_gradients = self.training_config.clip_gradients
        self.max_epochs = self.training_config.max_epochs

        self.early_stopping = EarlyStopping(
            self.model,
            self.checkpoint,
            early_stop_criteria,
            patience=early_stop_patience,
            minimize=early_stop_minimize,
            should_stop=early_stop_enabled,
        )
        self.current_epoch = 0
        self.current_iteration = 0
        self.num_updates = 0

        self.checkpoint.load_state_dict()

        self.not_debug = self.training_config.logger_level != "debug"

        self.lr_scheduler = None

        if self.training_config.lr_scheduler is True:
            self.lr_scheduler = build_scheduler(self.optimizer, self.config)

        self.tb_writer = None

        if self.training_config.tensorboard:
            log_dir = self.writer.log_folder
            env_tb_logdir = get_mmf_env(key="tensorboard_logdir")
            if env_tb_logdir:
                log_dir = env_tb_logdir

            self.tb_writer = TensorboardLogger(log_dir, self.current_iteration)

    def config_based_setup(self):
        seed = self.config.training.seed
        if seed is None:
            return

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self):
        self.writer.write("===== Model =====")
        self.writer.write(self.model)

        print_model_parameters(self.model)

        if "train" not in self.run_type:
            self.inference()
            return

        should_break = False

        if self.max_epochs is None:
            self.max_epochs = math.inf
        else:
            self.max_updates = self.max_epochs * len(self.train_loader)

        self.model.train()
        self.train_timer = Timer()
        self.snapshot_timer = Timer()

        self.profile("Setup Time")

        torch.autograd.set_detect_anomaly(True)
        self.writer.write("Starting training...")

        while self.num_updates < self.max_updates and not should_break:
            self.current_epoch += 1
            registry.register("current_epoch", self.current_epoch)

            # Seed the sampler in case if it is distributed
            self.dataset_loader.seed_sampler("train", self.current_epoch)

            if self.current_epoch > self.max_epochs:
                break

            for batch in self.train_loader:
                self.profile("Batch load time")
                self.current_iteration += 1
                self.writer.write(self.num_updates + 1, "debug")

                report = self._forward_pass(batch)
                loss = self._extract_loss(report)
                self._backward(loss)
                should_break = self._logistics(report)

                if self.num_updates > self.max_updates:
                    should_break = True

                if should_break:
                    break

            # In distributed, each worker will complete one epoch when we reach this
            # as each worker is an individual instance
            self.current_epoch += get_world_size() - 1
        self.finalize()

    def _run_scheduler(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.num_updates)

    def _forward_pass(self, batch):
        prepared_batch = self.dataset_loader.prepare_batch(batch)
        self.profile("Batch prepare time")
        # Arguments should be a dict at this point
        model_output = self.model(prepared_batch)
        report = Report(prepared_batch, model_output)
        self.profile("Forward time")

        return report

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        if self.should_clip_gradients:
            clip_gradients(self.model, self.num_updates, self.tb_writer, self.config)

        self.optimizer.step()
        self._run_scheduler()
        self.num_updates += 1
        self.profile("Backward time")

    def _extract_loss(self, report):
        loss_dict = report.losses
        loss = sum([loss.mean() for loss in loss_dict.values()])
        return loss

    def finalize(self):
        self.writer.write("Stepping into final validation check")

        # Only do when run_type has train as it shouldn't happen on validation and
        # inference runs. Inference will take care of this anyways. Also, don't run
        # if current iteration is divisble by snapshot interval as it will just
        # be a repeat
        if (
            "train" in self.run_type
            and self.num_updates % self.evaluation_interval != 0
        ):
            self._try_full_validation(force=True)

        self.checkpoint.restore()
        self.checkpoint.finalize()
        self.inference()

        self.writer.write(f"Finished run in {self.total_timer.get_time_since_start()}")

    def _update_meter(self, report, meter=None, eval_mode=False):
        if meter is None:
            meter = self.meter

        if hasattr(report, "metrics"):
            metrics_dict = report.metrics
            reduced_metrics_dict = reduce_dict(metrics_dict)

        if not eval_mode:
            loss_dict = report.losses
            reduced_loss_dict = reduce_dict(loss_dict)

        with torch.no_grad():
            # Add metrics to meter only when mode is `eval`
            meter_update_dict = {}
            if not eval_mode:
                loss_key = report.dataset_type + "/total_loss"
                reduced_loss = sum([loss.mean() for loss in reduced_loss_dict.values()])
                if hasattr(reduced_loss, "item"):
                    reduced_loss = reduced_loss.item()

                registry.register(loss_key, reduced_loss)
                meter_update_dict.update({loss_key: reduced_loss})
                meter_update_dict.update(reduced_loss_dict)
            if hasattr(report, "metrics"):
                meter_update_dict.update(reduced_metrics_dict)
            meter.update(meter_update_dict, report.batch_size)

    def _logistics(self, report):
        registry.register("current_iteration", self.current_iteration)
        registry.register("num_updates", self.num_updates)

        should_print = self.num_updates % self.log_interval == 0
        should_break = False
        extra = {}

        if should_print is True:
            if "cuda" in str(self.device):
                extra["max mem"] = torch.cuda.max_memory_allocated() / 1024
                extra["max mem"] //= 1024

            if self.training_config.experiment_name:
                extra["experiment"] = self.training_config.experiment_name

            extra.update(
                {
                    "epoch": self.current_epoch,
                    "num_updates": self.num_updates,
                    "iterations": self.current_iteration,
                    "max_updates": self.max_updates,
                    "lr": "{:.5f}".format(self.optimizer.param_groups[0]["lr"]).rstrip(
                        "0"
                    ),
                    "ups": "{:.2f}".format(
                        self.log_interval / self.train_timer.unix_time_since_start()
                    ),
                    "time": self.train_timer.get_time_since_start(),
                    "time_since_start": self.total_timer.get_time_since_start(),
                    "eta": self._calculate_time_left(),
                }
            )

            self.train_timer.reset()
            # Calculate metrics every log interval for debugging
            if self.training_config.evaluate_metrics:
                report.metrics = self.metrics(report, report)
            self._update_meter(report, self.meter)

            self._summarize_report(self.meter, should_print=should_print, extra=extra)

        self._try_snapshot()
        should_break = self._try_full_validation()

        return should_break

    def _try_snapshot(self):
        if self.num_updates % self.checkpoint_interval == 0:
            self.writer.write("Checkpoint time. Saving a checkpoint.")
            self.checkpoint.save(
                self.num_updates, self.current_iteration, update_best=False
            )

    def _try_full_validation(self, force=False):
        should_break = False

        if self.num_updates % self.evaluation_interval == 0 or force:
            self.snapshot_timer.reset()
            self.writer.write("Evaluation time. Running on full validation set...")
            # Validation and Early stopping
            # Create a new meter for this case
            report, meter = self.evaluate(self.val_loader)

            extra = {
                "num_updates": self.num_updates,
                "epoch": self.current_epoch,
                "iterations": self.current_iteration,
                "max_updates": self.max_updates,
                "val_time": self.snapshot_timer.get_time_since_start(),
            }

            stop = self.early_stopping(self.num_updates, self.current_iteration, meter)
            stop = bool(broadcast_scalar(stop, src=0, device=self.device))

            extra.update(self.early_stopping.get_info())

            self._summarize_report(meter, extra=extra)
            gc.collect()

            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

            if stop is True:
                self.writer.write("Early stopping activated")
                should_break = True

            self.train_timer.reset()

        return should_break

    def evaluate(self, loader, use_tqdm=False, single_batch=False):
        meter = Meter()

        with torch.no_grad():
            self.model.eval()
            disable_tqdm = not use_tqdm or not is_master()
            combined_report = None

            for batch in tqdm(loader, disable=disable_tqdm):
                report = self._forward_pass(batch)
                self._update_meter(report, meter)

                # accumulate necessary params for metric calculation
                if combined_report is None:
                    combined_report = report
                else:
                    combined_report.accumulate_tensor_fields(
                        report, self.metrics.required_params
                    )
                    combined_report.batch_size += report.batch_size

                if single_batch is True:
                    break

            combined_report.metrics = self.metrics(combined_report, combined_report)
            self._update_meter(combined_report, meter, eval_mode=True)

            self.model.train()

        return combined_report, meter

    def _summarize_report(self, meter, should_print=True, extra=None):
        if extra is None:
            extra = {}
        if not is_master():
            return

        if self.training_config.tensorboard:
            scalar_dict = meter.get_scalar_dict()
            self.tb_writer.add_scalars(scalar_dict, self.current_iteration)

        if not should_print:
            return
        log_dict = {"progress": f"{self.num_updates}/{self.max_updates}"}
        log_dict.update(meter.get_log_dict())
        log_dict.update(extra)

        self.writer.log_progress(log_dict)

    def inference(self):
        if "val" in self.run_type:
            self._inference_run("val")

        if any(rt in self.run_type for rt in ["inference", "test", "predict"]):
            self._inference_run("test")

    def _inference_run(self, dataset_type):
        if self.config.evaluation.predict:
            self.predict(dataset_type)
            return

        self.writer.write(f"Starting inference on {dataset_type} set")

        report, meter = self.evaluate(
            getattr(self, f"{dataset_type}_loader"), use_tqdm=True
        )
        prefix = f"{report.dataset_name}: full {dataset_type}"
        self._summarize_report(meter, prefix)

    def _calculate_time_left(self):
        time_taken_for_log = time.time() * 1000 - self.train_timer.start
        iterations_left = self.max_updates - self.num_updates
        num_logs_left = iterations_left / self.log_interval
        time_left = num_logs_left * time_taken_for_log

        snapshot_iteration = self.snapshot_iterations / self.log_interval
        snapshot_iteration *= iterations_left / self.evaluation_interval
        time_left += snapshot_iteration * time_taken_for_log

        return self.train_timer.get_time_hhmmss(gap=time_left)

    def profile(self, text):
        if self.not_debug:
            return
        self.writer.write(text + ": " + self.profiler.get_time_since_start(), "debug")
        self.profiler.reset()

    def predict(self, dataset_type):
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            message = f"Starting {dataset_type} inference predictions"
            self.writer.write(message)

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm(dataloader):
                    prepared_batch = reporter.prepare_batch(batch)
                    model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report, self.model)

            self.writer.write("Finished predicting")
            self.model.train()
