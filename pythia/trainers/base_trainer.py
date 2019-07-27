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
from pythia.common.task_loader import TaskLoader
from pythia.utils.build_utils import build_model, build_optimizer
from pythia.utils.checkpoint import Checkpoint
from pythia.utils.distributed_utils import (broadcast_scalar, is_main_process,
                                            reduce_dict, synchronize)
from pythia.utils.early_stopping import EarlyStopping
from pythia.utils.general import clip_gradients, lr_lambda_update
from pythia.utils.logger import Logger
from pythia.utils.timer import Timer


@registry.register_trainer('base_trainer')
class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.profiler = Timer()

    def load(self):
        self._init_process_group()

        self.run_type = self.config.training_parameters.get("run_type", "train")
        self.task_loader = TaskLoader(self.config)

        self.writer = Logger(self.config)
        registry.register("writer", self.writer)

        self.configuration = registry.get("configuration")
        self.configuration.pretty_print()

        self.config_based_setup()

        self.load_task()
        self.load_model()
        self.load_optimizer()
        self.load_extras()

    def _init_process_group(self):
        training_parameters = self.config.training_parameters
        self.local_rank = training_parameters.local_rank
        self.device = training_parameters.device

        if self.local_rank is not None and training_parameters.distributed:
            if not torch.distributed.is_nccl_available():
                raise RuntimeError(
                    "Unable to initialize process group: NCCL is not available"
                )
            torch.distributed.init_process_group(backend="nccl")
            synchronize()

        if (
            "cuda" in self.device
            and training_parameters.distributed
            and self.local_rank is not None
        ):
            self.device = torch.device("cuda", self.local_rank)

        registry.register("current_device", self.device)

    def load_task(self):
        self.writer.write("Loading tasks and data", "info")
        self.task_loader.load_task()

        self.task_loader.make_dataloaders()

        self.train_loader = self.task_loader.train_loader
        self.val_loader = self.task_loader.val_loader
        self.test_loader = self.task_loader.test_loader
        self.train_task = self.task_loader.train_task
        self.val_task = self.task_loader.val_task

        # Total iterations for snapshot
        self.snapshot_iterations = len(self.val_task)
        self.snapshot_iterations //= self.config.training_parameters.batch_size

        self.test_task = self.task_loader.test_task

    def load_model(self):
        attributes = self.config.model_attributes[self.config.model]
        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_attributes[attributes]

        attributes["model"] = self.config.model

        self.task_loader.update_registry_for_model(attributes)
        self.model = build_model(attributes)
        self.task_loader.clean_config(attributes)
        training_parameters = self.config.training_parameters

        data_parallel = training_parameters.data_parallel
        distributed = training_parameters.distributed

        registry.register("data_parallel", data_parallel)
        registry.register("distributed", distributed)

        if "cuda" in str(self.config.training_parameters.device):
            rank = self.local_rank if self.local_rank is not None else 0
            device_info = "CUDA Device {} is: {}".format(
                rank, torch.cuda.get_device_name(self.local_rank)
            )

            self.writer.write(device_info, log_all=True)

        self.model = self.model.to(self.device)

        self.writer.write("Torch version is: " + torch.__version__)

        if (
            "cuda" in str(self.device)
            and torch.cuda.device_count() > 1
            and data_parallel is True
        ):
            self.model = torch.nn.DataParallel(self.model)

        if (
            "cuda" in str(self.device)
            and self.local_rank is not None
            and distributed is True
        ):
            torch.cuda.set_device(self.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank]
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
        self.snapshot_interval = self.training_parameters.snapshot_interval
        self.max_iterations = self.training_parameters.max_iterations
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
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self):
        self.writer.write("===== Model =====")
        self.writer.write(self.model)

        if "train" not in self.run_type:
            self.inference()
            return

        should_break = False

        if self.max_epochs is None:
            self.max_epochs = math.inf
        else:
            self.max_iterations = math.inf

        self.model.train()
        self.train_timer = Timer()
        self.snapshot_timer = Timer()

        self.profile("Setup Time")

        torch.autograd.set_detect_anomaly(True)

        self.writer.write("Starting training...")
        while self.current_iteration < self.max_iterations and not should_break:
            self.current_epoch += 1
            registry.register("current_epoch", self.current_epoch)

            # Seed the sampler in case if it is distributed
            self.task_loader.seed_sampler("train", self.current_epoch)

            if self.current_epoch > self.max_epochs:
                break

            for batch in self.train_loader:
                self.profile("Batch load time")
                self.current_iteration += 1
                self.writer.write(self.current_iteration, "debug")

                registry.register("current_iteration", self.current_iteration)

                if self.current_iteration > self.max_iterations:
                    break

                self._run_scheduler()
                report = self._forward_pass(batch)
                self._update_meter(report, self.meter)
                loss = self._extract_loss(report)
                self._backward(loss)
                should_break = self._logistics(report)

                if should_break:
                    break

        self.finalize()

    def _run_scheduler(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.current_iteration)

    def _forward_pass(self, batch):
        prepared_batch = self.task_loader.prepare_batch(batch)
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
        self.profile("Backward time")

    def _extract_loss(self, report):
        loss_dict = report.losses
        loss = sum([loss.mean() for loss in loss_dict.values()])
        return loss

    def finalize(self):
        self.writer.write("Stepping into final validation check")
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

            extra.update(
                {
                    "lr": "{:.5f}".format(self.optimizer.param_groups[0]["lr"]).rstrip(
                        "0"
                    ),
                    "time": self.train_timer.get_time_since_start(),
                    "eta": self._calculate_time_left(),
                }
            )

            self.train_timer.reset()

            _, meter = self.evaluate(self.val_loader, single_batch=True)
            self.meter.update_from_meter(meter)

        # Don't print train metrics if it is not log interval
        # so as to escape clutter
        self._summarize_report(
            self.meter,
            should_print=should_print,
            extra=extra,
            prefix=report.dataset_name,
        )

        should_break = self._try_full_validation()

        return should_break

    def _try_full_validation(self, force=False):
        should_break = False

        if self.current_iteration % self.snapshot_interval == 0 or force:
            self.writer.write("Evaluation time. Running on full validation set...")
            # Validation and Early stopping
            # Create a new meter for this case
            report, meter = self.evaluate(self.val_loader)

            extra = {"validation time": self.snapshot_timer.get_time_since_start()}

            stop = self.early_stopping(self.current_iteration, meter)
            stop = bool(broadcast_scalar(stop, src=0, device=self.device))

            extra.update(self.early_stopping.get_info())

            prefix = "{}: full val".format(report.dataset_name)

            self._summarize_report(meter, prefix=prefix, extra=extra)
            self.snapshot_timer.reset()
            gc.collect()

            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

            if stop is True:
                self.writer.write("Early stopping activated")
                should_break = True

        return should_break

    def evaluate(self, loader, use_tqdm=False, single_batch=False):
        meter = Meter()

        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(loader, disable=not use_tqdm):
                report = self._forward_pass(batch)
                self._update_meter(report, meter, eval_mode=True)

                if single_batch is True:
                    break
            self.model.train()

        return report, meter

    def _summarize_report(self, meter, prefix="", should_print=True, extra={}):
        if not is_main_process():
            return

        scalar_dict = meter.get_scalar_dict()
        self.writer.add_scalars(scalar_dict, registry.get("current_iteration"))

        if not should_print:
            return

        print_str = []

        if len(prefix):
            print_str += [prefix + ":"]

        print_str += ["{}/{}".format(self.current_iteration, self.max_iterations)]
        print_str += [str(meter)]
        print_str += ["{}: {}".format(key, value) for key, value in extra.items()]

        self.writer.write(meter.delimiter.join(print_str))

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
        iterations_left = self.max_iterations - self.current_iteration
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
        reporter = self.task_loader.get_test_reporter(dataset_type)
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
