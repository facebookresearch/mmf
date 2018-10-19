import math
import torch
import gc
import sys

from torch import optim

from pythia.utils.flags import flags
from pythia.utils.configuration import Configuration
from pythia.utils.checkpoint import Checkpoint
from pythia.utils.logger import Logger
from pythia.utils.general import lr_lambda_update, clip_gradients, \
                get_optimizer_parameters, dict_to_string
from pythia.utils.build import build_model
from pythia.utils.timer import Timer
from pythia.utils.early_stopping import EarlyStopping
from pythia.core.task_loader import TaskLoader
from pythia.core.registry import Registry


class Trainer:
    def __init__(self):
        parser = flags.get_parser()
        self.args = parser.parse_args()

    def load(self):
        self.load_config()
        self.load_task()

        self.writer = Logger(self.config)
        Registry.register('writer', self.writer)

        self.load_model()
        self.load_optimizer()
        self.load_extras()

    def load_config(self):
        # TODO: Review configuration update once again
        # (remember clip_gradients case)
        self.configuration = Configuration(self.args.config)
        self.args.use_cuda = not self.args.no_cuda
        self.configuration.update_with_args(self.args, force=True)

        self.config = self.configuration.get_config()
        Registry.register('config', self.config)

        self.config_based_setup()

    def load_task(self):
        self.task_loader = TaskLoader(self.config)
        self.task_loader.load_config()

        self.configuration.update_with_task_config(
            self.task_loader.get_config()
        )

        # Update with the config override if passed
        self.configuration.override_with_cmd_config(self.args.config_override)

        # Update with args once again as they are the most important
        self.configuration.update_with_args(self.args)
        self.configuration.pretty_print()

        self.task_loader.load_task()

        self.task_loader.make_dataloaders()

        self.train_loader = self.task_loader.train_loader
        self.dev_loader = self.task_loader.dev_loader
        self.test_loader = self.task_loader.test_loader
        self.train_task = self.task_loader.train_task
        self.dev_task = self.task_loader.dev_task
        self.test_task = self.task_loader.test_task

    def load_model(self):
        attributes = self.config['model_attributes'][self.config['model']]
        attributes['model'] = self.config['model']

        data_root_dir = self.config['training_parameters']['data_root_dir']
        attributes['data_root_dir'] = data_root_dir

        self.task_loader.update_config_for_model(attributes)
        self.model = build_model(attributes)
        self.task_loader.clean_config(attributes)

        use_cuda = self.config['training_parameters']['use_cuda']
        if use_cuda:
            self.model = self.model.cuda()

        if use_cuda and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def load_optimizer(self):
        optimizer_method = self.config['optimizer_attributes']['type']
        optimizer_class = getattr(optim, optimizer_method)

        parameters = get_optimizer_parameters(self.model, self.config)
        rest_optimizer_params = self.config['optimizer_attributes']['params']
        self.optimizer = optimizer_class(parameters, **rest_optimizer_params)

    def load_extras(self):
        self.checkpoint = Checkpoint(self)

        training_parameters = self.config['training_parameters']

        monitored_metric = training_parameters['monitored_metric']
        metric_minimize = training_parameters['metric_minimize']
        should_early_stop = training_parameters['should_early_stop']
        patience = training_parameters['patience']

        self.early_stopping = EarlyStopping(self.model,
                                            self.checkpoint,
                                            self.dev_meter,
                                            monitored_metric,
                                            patience=patience,
                                            minimize=metric_minimize,
                                            should_stop=should_early_stop)

        self.current_epoch = 0
        self.current_iteration = 0

        self.checkpoint.load_state_dict()

        self.lr_scheduler = None
        if training_parameters['lr_scheduler'] is True:
            scheduler_class = optim.lr_scheduler.LambdaLR
            scheduler_func = lambda x: lr_lambda_update(x, self.config)
            self.lr_scheduler = scheduler_class(self.optimizer,
                                                lr_lambda=scheduler_func)

    def config_based_setup(self):
        torch.manual_seed(self.config['seed'])

        if self.config['training_parameters']['use_cuda']:
            torch.cuda.manual_seed(self.config['seed'])

    def train(self):
        training_parameters = self.config['training_parameters']
        log_interval = training_parameters['log_interval']
        snapshot_interval = training_parameters['snapshot_interval']
        max_iterations = training_parameters['max_iterations']
        should_clip_gradients = training_parameters['clip_gradients']
        max_epochs = self.config['max_epochs']

        self.writer.write(self.model)
        should_check_on_epoch = False

        if max_epochs is not None:
            should_check_on_epoch = True
            max_iterations = math.inf

        self.model.train()
        self.train_timer = Timer()
        self.snapshot_timer = Timer()

        while self.current_iteration < max_iterations:
            self.current_epoch += 1
            Registry.register('current_epoch', self.current_epoch)

            if should_check_on_epoch and self.current_epoch > max_epochs:
                break

            for batch in self.train_loader:
                self.current_iteration += 1
                Registry.register('current_iteration', self.current_iteration)

                if self.current_iteration > max_iterations:
                    break

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(self.current_iteration)

                self.optimizer.zero_grad()

                data, y = self.task_loader.prepare_batch('train', batch)

                # Arguments should be a dict at this point
                output = self.model(**data)
                loss = self.task_loader.calculate_loss('train', output, y)

                loss.backward()

                if should_clip_gradients:
                    clip_gradients(self.model, self.current_iteration,
                                   self.writer, self.config)

                self.optimizer.step()

                extra_info = None
                should_print = self.current_iteration % log_interval == 0

                # TODO: Move in separate function
                if should_print is True:
                    sys.stdout.flush()
                    extra_info = self.single_batch_eval('dev', self.dev_loader)
                    time_taken = self.train_timer.get_time_since_start()
                    extra_info += ", time: %s" % time_taken

                # Don't print train metrics if it is not log interval
                # so as to escape clutter
                self.task_loader.report_metrics('train', loss.data.item(),
                                                extra_info=extra_info,
                                                should_print=should_print)

                if should_print is True:
                    self.train_timer.reset()

                if self.current_iteration % snapshot_interval == 0:
                    # TODO: Implement early stopping
                    # TODO: Do validation check here
                    avg_loss = self.evaluate('dev', self.dev_loader)
                    time_taken = self.snapshot_timer.get_time_since_start()
                    extra_info = ", time: %s" % time_taken
                    self.snapshot_timer.reset()
                    stop = self.early_stopping(self.current_iteration,
                                               avg_loss)
                    extra_info += "\n %s" % self.early_stopping.get_info()

                    self.task_loader.report_metrics('dev', avg_loss,
                                                    extra_info=extra_info)

                    if stop is True:
                        self.writer.write("Early stopping activated")
                        break

                gc.collect()

        avg_test_loss = self.evaluate('test', self.test_loader)
        self.task_loader.report_metrics('test', avg_test_loss)

    def single_batch_eval(self, dataset_type, loader):
        self.model.eval()
        batch = next(iter(loader))
        self.task_loader.reset_meters(dataset_type)

        data, y = self.task_loader.prepare_batch(loader.dataset_type, batch)
        output = self.model(**data)
        self.task_loader.calculate_loss(dataset_type, output, y)
        self.model.train()

        # TODO: Do replace in log string function itself
        return "Dev: " + dict_to_string(Registry.get('metrics.%s' %
                                                     dataset_type))

    def evaluate(self, dataset_type, loader):
        self.model.eval()
        self.task_loader.reset_meters(dataset_type)

        total_loss = 0
        total_samples = 0

        for batch in loader:
            data, y = self.task_loader.prepare_batch(dataset_type,
                                                     batch)
            total_samples += y.size(0)

            output = self.model(**data)

            loss = self.task_loader.calculate_loss(dataset_type, output, y)
            if loss is not None:
                total_loss += loss.data.item() * y.size(0)

        self.model.train()
        return total_loss / total_samples
