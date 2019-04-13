import math
import torch
import gc

from torch import optim
from tqdm import tqdm

from pythia.utils.flags import flags
from pythia.utils.configuration import Configuration
from pythia.utils.checkpoint import Checkpoint
from pythia.utils.logger import Logger
from pythia.utils.general import lr_lambda_update, clip_gradients, \
                get_optimizer_parameters, dict_to_string, get_current_tensors
from pythia.utils.build import build_model
from pythia.utils.distributed_utils import reduce_tensor, synchronize
from pythia.utils.timer import Timer
from pythia.utils.early_stopping import EarlyStopping
from pythia.common.task_loader import TaskLoader
from pythia.common.registry import registry
from pythia.common.report import Report


class Trainer:
    def __init__(self, args, *rest, **kwargs):
        self.args = args
        self.profiler = Timer()

    def load(self):
        self.load_config()
        self._init_process_group()

        self.run_type = self.config.training_parameters.get('run_type',
                                                            "train")
        self.task_loader = TaskLoader(self.config)

        self.writer = Logger(self.config)
        registry.register('writer', self.writer)

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
                raise RuntimeError("Unable to initialize process group: "
                                   "NCCL is not available")
            torch.distributed.init_process_group(backend="nccl")
            synchronize()

        if "cuda" in self.device and training_parameters.distributed \
            and self.local_rank is not None:
            self.device = torch.device("cuda", self.local_rank)

        registry.register("current_device", self.device)

    def load_config(self):
        # TODO: Review configuration update once again
        # (remember clip_gradients case)
        self.configuration = Configuration(self.args.config)

        # Update with the config override if passed
        self.configuration.override_with_cmd_config(self.args.config_override)

        # Now, update with opts args that were passed
        self.configuration.override_with_cmd_opts(self.args.opts)

        # Finally, update with args that were specifically passed
        # as arguments
        self.configuration.update_with_args(self.args)
        self.configuration.freeze()

        self.config = self.configuration.get_config()
        registry.register('config', self.config)

    def load_task(self):
        self.writer.write("Loading tasks and data", "info")
        self.task_loader.load_task()

        self.task_loader.make_dataloaders()

        self.train_loader = self.task_loader.train_loader
        self.val_loader = self.task_loader.val_loader
        self.test_loader = self.task_loader.test_loader
        self.train_task = self.task_loader.train_task
        self.val_task = self.task_loader.val_task
        self.test_task = self.task_loader.test_task
        self.test_reporter = self.task_loader.test_reporter

    def load_model(self):
        attributes = self.config.model_attributes[self.config.model]
        # Easy way to point to config for other model
        if type(attributes) == str:
            attributes = self.config.model_attributes[attributes]

        attributes['model'] = self.config.model

        self.task_loader.update_registry_for_model(attributes)
        self.model = build_model(attributes)
        self.task_loader.clean_config(attributes)
        training_parameters = self.config.training_parameters

        data_parallel = training_parameters.data_parallel
        distributed = training_parameters.distributed

        registry.register('data_parallel', data_parallel)
        registry.register('distributed', distributed)

        if "cuda" in str(self.config.training_parameters.device):
            rank = self.local_rank if self.local_rank is not None else 0
            self.writer.write("CUDA Device {} is: {}".format(
                rank, torch.cuda.get_device_name(self.local_rank)
            ))

        self.model = self.model.to(self.device)

        self.writer.write("Torch version is: " + torch.__version__)

        if "cuda" in str(self.device) and torch.cuda.device_count() > 1 \
           and data_parallel is True:
            self.model = torch.nn.DataParallel(self.model)

        if "cuda" in str(self.device) and self.local_rank is not None \
            and distributed is True:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )

    def load_optimizer(self):
        optimizer_method = self.config.optimizer_attributes.type
        optimizer_class = getattr(optim, optimizer_method)

        # TODO: Allow custom optimizer making
        parameters = get_optimizer_parameters(self.model, self.config)
        rest_optimizer_params = self.config.optimizer_attributes.params
        self.optimizer = optimizer_class(parameters, **rest_optimizer_params)

    def load_extras(self):
        self.checkpoint = Checkpoint(self)

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

        self.early_stopping = EarlyStopping(self.model,
                                            self.checkpoint,
                                            monitored_metric,
                                            patience=patience,
                                            minimize=metric_minimize,
                                            should_stop=should_early_stop)
        self.current_epoch = 0
        self.current_iteration = 0

        self.checkpoint.load_state_dict()

        self.not_debug = self.training_parameters.logger_level != "debug"

        self.lr_scheduler = None

        # TODO: Allow custom scheduler
        if self.training_parameters.lr_scheduler is True:
            scheduler_class = optim.lr_scheduler.LambdaLR
            scheduler_func = lambda x: lr_lambda_update(x, self.config)
            self.lr_scheduler = scheduler_class(self.optimizer,
                                                lr_lambda=scheduler_func)

    def config_based_setup(self):
        seed = self.config.training_parameters.seed
        if self.local_rank is not None:
            seed += self.local_rank

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def train(self):
        if "train" not in self.run_type:
            self.predict()
            return

        self.writer.write("===== Model =====")
        self.writer.write(self.model)

        should_break = False

        if self.max_epochs is None:
            self.max_epochs = math.inf
        else:
            self.max_iterations = math.inf

        self.model.train()
        self.train_timer = Timer()
        self.snapshot_timer = Timer()

        self.task_loader.reset_meters("train")

        self.profile("Setup Time")

        torch.autograd.set_detect_anomaly(True)

        self.writer.write("Starting training...")
        while self.current_iteration < self.max_iterations and not should_break:
            self.current_epoch += 1
            registry.register('current_epoch', self.current_epoch)

            if self.current_epoch > self.max_epochs:
                break

            for batch in self.train_loader:
                self.profile("Batch load time")
                self.current_iteration += 1
                self.writer.write(self.current_iteration, "debug")

                registry.register('current_iteration', self.current_iteration)

                if self.current_iteration > self.max_iterations:
                    break

                report = self._forward_pass(batch)
                should_break = self._logistics(report)

                if should_break:
                    break

        self.checkpoint.restore()
        self.predict()

    def _forward_pass(self, batch, eval_mode=False):
        prepared_batch = self.task_loader.prepare_batch(batch)
        self.profile("Batch prepare time")

        # Arguments should be a dict at this point
        model_output = self.model(prepared_batch)
        report = Report(batch, prepared_batch, model_output)


        self.task_loader.verbose_dump(report)
        loss = self.task_loader.calculate_loss_and_metrics(report)
        self.profile("Forward time")

        reduced_loss = reduce_tensor(loss)

        if not eval_mode:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.profile("Backward time")

        report.loss = reduced_loss

        return report

    def _logistics(self, report):
        extra_info = None

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.current_iteration)

        if self.should_clip_gradients:
            clip_gradients(self.model, self.current_iteration,
                           self.writer, self.config)

        should_print = self.current_iteration % self.log_interval == 0
        should_break = False
        if should_print is True:
            self.evaluate('val', self.val_loader, single_batch=True)
            extra_info = "val: " + dict_to_string(registry.get('metrics.val'))
            time_taken = self.train_timer.get_time_since_start()
            extra_info += ", time: %s" % time_taken

        # Don't print train metrics if it is not log interval
        # so as to escape clutter
        self.task_loader.report_metrics(
            'train', report, report.loss.item(),
            extra_info=extra_info, should_print=should_print
        )

        if should_print is True:
            self.train_timer.reset()

        # TODO: Move to separate function
        if self.current_iteration % self.snapshot_interval == 0:
            # Validation and Early stopping
            avg_loss = self.evaluate('val', self.val_loader)

            time_taken = self.snapshot_timer.get_time_since_start()
            extra_info = ", time: %s" % time_taken
            self.snapshot_timer.reset()

            stop = self.early_stopping(self.current_iteration)
            extra_info += "\n%s" % self.early_stopping.get_info()

            self.task_loader.report_metrics(
                'val', None, avg_loss, extra_info=extra_info
            )
            gc.collect()

            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

            if stop is True:
                self.writer.write("Early stopping activated")
                should_break = True

        return should_break

    def evaluate(self, dataset_type, loader, use_tqdm=False,
                 single_batch=False):
        self.task_loader.reset_meters(dataset_type)

        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(loader, disable=not use_tqdm):
                report = self._forward_pass(batch, eval_mode=True)
                if report.loss is not None:
                    total_loss += report.loss.item() * \
                        report.batch_size

                total_samples += report.batch_size

                if single_batch is True:
                    break

            self.model.train()

        return total_loss / total_samples

    def predict(self):
        if "predict" not in self.run_type:
            return

        # TODO: Make evalai based prediction dataset dependent i.e.
        # it will become a config parameter for dataset
        if self.test_reporter is not None:
            self.predict_for_evalai()
        else:
            self.writer.write("Starting predictions")

            avg_test_loss = self.evaluate('test', self.test_loader,
                                          use_tqdm=True)
            self.task_loader.report_metrics('test', None, avg_test_loss)

    def profile(self, text):
        if self.not_debug:
            return
        self.writer.write(text + ": " + self.profiler.get_time_since_start(),
                          "debug")
        self.profiler.reset()

    def predict_for_evalai(self):
        with torch.no_grad():
            self.model.eval()
            self.writer.write("Starting prediction for evalai")

            while self.test_reporter.next_dataset():
                dataloader = self.test_reporter.get_dataloader()

                for batch in tqdm(dataloader):
                    prepared_batch = self.test_reporter.prepare_batch(batch)
                    model_output = self.model(prepared_batch)
                    report = Report(batch, prepared_batch, model_output)

                    self.test_reporter.add_to_report(report)

            self.writer.write("Finished predicting")
            self.model.train()
