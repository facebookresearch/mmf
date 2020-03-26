# Copyright (c) Facebook, Inc. and its affiliates.
import gc
import math
import time
import random

import torch
from torch import optim
from tqdm import tqdm

from pythia.common.meter import Meter
from pythia.common.registry import registry
from pythia.common.report import Report
from pythia.common.dataset_loader import DatasetLoader
from pythia.utils.build_utils import build_model, build_optimizer
from pythia.utils.checkpoint import Checkpoint
from pythia.utils.distributed_utils import (broadcast_scalar, is_master,
                                            reduce_dict, synchronize, distributed_init)
from pythia.utils.early_stopping import EarlyStopping
from pythia.utils.general import clip_gradients, print_model_parameters
from pythia.utils.timer import Timer


@registry.register_trainer('base_trainer')
class BaseTrainer:
    def __init__(self, configuration):
        self.configuration = configuration
        self.config = self.configuration.config
        self.profiler = Timer()
        if self.configuration is not None:
            self.args = self.configuration.args

    def load(self):
        self._set_device()

        self.run_type = self.config.training_parameters.get("run_type", "train")
        self.dataset_loader = DatasetLoader(self.config)
        self._datasets = self.config.datasets

        self.writer = Logger(self.config)
        registry.register("writer", self.writer)

        self.configuration.pretty_print()

        self.config_based_setup()

        self.load_task()
        self.load_model()
        self.load_optimizer()
        self.load_extras()

    def _set_device(self):
        self.local_rank = self.args.device_id
        self.device = self.local_rank
        self.distributed = False

        if self.args.distributed_init_method is not None:
            self.distributed = True
            self.device = torch.device("cuda", self.local_rank)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        registry.register("current_device", self.device)

    def load_task(self):
        self.writer.write("Loading datasets", "info")
        self.dataset_loader.load_datasets()

        self.train_dataset = self.dataset_loader.train_dataset
        self.val_dataset = self.dataset_loader.val_dataset

        # Total iterations for snapshot
        self.snapshot_iterations = len(self.val_dataset)
        self.snapshot_iterations //= self.config.training_parameters.batch_size

        self.test_dataset = self.dataset_loader.test_dataset

        self.train_loader = self.dataset_loader.train_loader
        self.val_loader = self.dataset_loader.val_loader
        self.test_loader = self.dataset_loader.test_loader

    def load_model(self):
        attributes = self.config.model_attributes[self.config.model]
        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_attributes[attributes]

        attributes["model"] = self.config.model

        self.dataset_loader.update_registry_for_model(attributes)
        self.model = build_model(attributes)
        self.dataset_loader.clean_config(attributes)
        training_parameters = self.config.training_parameters

        if "cuda" in str(self.device):
            rank = self.local_rank if self.local_rank is not None else 0
            device_info = "CUDA Device {} is: {}".format(
                self.args.distributed_rank, torch.cuda.get_device_name(self.local_rank)
            )

            self.writer.write(device_info, log_all=True)

        self.model = self.model.to(self.device)

        self.writer.write("Torch version is: " + torch.__version__)

        registry.register("data_parallel", False)
        registry.register("distributed", False)
        if (
            "cuda" in str(self.device)
            and torch.cuda.device_count() > 1
            and not self.distributed
        ):
            registry.register("data_parallel", True)
            self.model = torch.nn.DataParallel(self.model)

        if (
            "cuda" in str(self.device)
            and self.distributed
        ):
            registry.register("distributed", True)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank], output_device=self.local_rank,
                check_reduction=True,
                find_unused_parameters=training_parameters.find_unused_parameters
            )

    def load_optimizer(self):
        self.optimizer = build_optimizer(self.model, self.config)

    def load_extras(self):
        self.checkpoint = Checkpoint(self)
        self.meter = Meter()

        self.training_parameters = self.config.training_parameters

        monitored_metric = self.training_parameters.monitored_metric
        metric_minimize = self.training_parameters.metric_minimize
        should_early_stop = self.training_parameters.should_early_stop
        patience = self.training_parameters.patience

        self.log_interval = self.training_parameters.log_interval
        self.max_updates = self.training_parameters.max_updates
        self.should_clip_gradients = self.training_parameters.clip_gradients
        self.max_epochs = self.training_parameters.max_epochs

        self.early_stopping = EarlyStopping(
            self.model,
            self.checkpoint,
            monitored_metric,
            patience=patience,
            minimize=metric_minimize,
            should_stop=should_early_stop,
        )
        self.current_epoch = 0
        self.current_iteration = 0
        self.num_updates = 0

        self.checkpoint.load_state_dict()

        self.not_debug = self.training_parameters.logger_level != "debug"

        self.lr_scheduler = None

        # TODO: Allow custom scheduler
        if self.training_parameters.lr_scheduler is True:
            scheduler_class = optim.lr_scheduler.LambdaLR
            scheduler_func = lambda x: lr_lambda_update(x, self.config)
            self.lr_scheduler = scheduler_class(
                self.optimizer, lr_lambda=scheduler_func
            )

    def config_based_setup(self):
        seed = self.config.training_parameters.seed
        if seed is None:
            return

        random.seed(seed)
        torch.manual_seed(seed)
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
            self.max_updates = math.inf

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
            clip_gradients(self.model, self.current_iteration, self.writer, self.config)

        self.optimizer.step()
        self._run_scheduler()
        self.num_updates += 1
        self.profile("Backward time")

        self.profile("Backward time")

    def _extract_loss(self, report):
        loss_dict = report.losses
        loss = sum([loss.mean() for loss in loss_dict.values()])
        return loss

    def finalize(self):
        self.writer.write("Stepping into final validation check")

        # Only do when run_type has train as it shouldn't happen on validation and inference runs
        # Inference will take care of this anyways. Also, don't run if current iteration
        # is divisble by snapshot interval as it will just be a repeat
        if "train" in self.run_type and self.current_iteration % self.snapshot_interval != 0:
            self._try_full_validation(force=True)

        self.checkpoint.restore()
        self.checkpoint.finalize()
        self.inference()

    def _update_meter(self, report, meter=None, eval_mode=False):
        if meter is None:
            meter = self.meter

        loss_dict = report.losses
        metrics_dict = report.metrics

        reduced_loss_dict = reduce_dict(loss_dict)
        reduced_metrics_dict = reduce_dict(metrics_dict)

        loss_key = report.dataset_type + "/total_loss"

        with torch.no_grad():
            reduced_loss = sum([loss.mean() for loss in reduced_loss_dict.values()])
            if hasattr(reduced_loss, "item"):
                reduced_loss = reduced_loss.item()

            registry.register(loss_key, reduced_loss)

            meter_update_dict = {loss_key: reduced_loss}
            meter_update_dict.update(reduced_loss_dict)
            meter_update_dict.update(reduced_metrics_dict)
            meter.update(meter_update_dict)

    def _logistics(self, report):
        should_print = self.current_iteration % self.log_interval == 0
        should_break = False
        extra = {}

        if should_print is True:
            if "cuda" in str(self.device):
                extra["max mem"] = torch.cuda.max_memory_allocated() / 1024
                extra["max mem"] //= 1024

            if self.training_parameters.experiment_name:
                extra["experiment"] = self.training_parameters.experiment_name

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
                    "eta": self._calculate_time_left(),
                }
            )

            self.train_timer.reset()
            self._update_meter(report, self.meter)

        # Don't print train metrics if it is not log interval
        # so as to escape clutter
        self._summarize_report(
            self.meter,
            should_print=should_print,
                extra=extra
        )

        should_break = self._try_full_validation()

        return should_break

    def _try_full_validation(self, force=False):
        should_break = False

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
                "val_time": self.snapshot_timer.get_time_since_start()
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

            for batch in tqdm(loader, disable=disable_tqdm):
                report = self._forward_pass(batch)
                self._update_meter(report, meter, eval_mode=True)

                if single_batch is True:
                    break
            self.model.train()

        return report, meter

        if not is_master():
            return

        scalar_dict = meter.get_scalar_dict()
        self.writer.add_scalars(scalar_dict, registry.get("current_iteration"))

        if not should_print:
            return
        log_dict = {
            "progress": "{}/{}".format(self.num_updates, self.max_updates)
        }
        log_dict.update(meter.get_log_dict())
        log_dict.update(extra)

        self.writer.log_progress(log_dict)

    def inference(self):
        if "val" in self.run_type:
            self._inference_run("val")

        if "inference" in self.run_type or "predict" in self.run_type:
            self._inference_run("test")

    def _inference_run(self, dataset_type):
        if self.config.training_parameters.evalai_inference is True:
            self.predict_for_evalai(dataset_type)
            return

        self.writer.write("Starting inference on {} set".format(dataset_type))

        report, meter = self.evaluate(
            getattr(self, "{}_loader".format(dataset_type)), use_tqdm=True
        )
        prefix = "{}: full {}".format(report.dataset_name, dataset_type)
        self._summarize_report(meter, prefix)

    def _calculate_time_left(self):
        time_taken_for_log = time.time() * 1000 - self.train_timer.start
        iterations_left = self.max_updates - self.num_updates
        num_logs_left = iterations_left / self.log_interval
        time_left = num_logs_left * time_taken_for_log

        snapshot_iteration = self.snapshot_iterations / self.log_interval
        snapshot_iteration *= iterations_left / self.snapshot_interval
        time_left += snapshot_iteration * time_taken_for_log

        return self.train_timer.get_time_hhmmss(gap=time_left)

    def profile(self, text):
        if self.not_debug:
            return
        self.writer.write(text + ": " + self.profiler.get_time_since_start(), "debug")
        self.profiler.reset()

    def predict_for_evalai(self, dataset_type):
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            message = "Starting {} inference for evalai".format(dataset_type)
            self.writer.write(message)

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm(dataloader):
                    prepared_batch = reporter.prepare_batch(batch)
                    model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report)

            self.writer.write("Finished predicting")
            self.model.train()
