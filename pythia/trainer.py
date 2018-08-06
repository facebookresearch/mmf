import math
import torch

from torch import optim

from pythia.utils.flags import flags
from pythia.utils.configuration import Configuration
from pythia.utils.logger import Logger
from pythia.utils.general import lr_lambda_update, clip_gradients
from pythia.utils.build import build_model
from pythia.task_loader import TaskLoader
from pythia.modules.losses import Loss


class Trainer:
    def __init__(self):
        parser = flags.get_parser()
        self.args = parser.parse_args()

    def load(self):
        self.load_config()
        self.load_task()

        self.writer = Logger(self.config)

        self.load_model()
        self.load_optimizer()
        self.load_extras()

    def load_config(self):
        self.configuration = Configuration(self.args.config)
        self.configuration.update_with_args(self.args)

        self.config = self.configuration.get_config()

        self.config_based_setup()

    def load_task(self):
        self.task_loader = TaskLoader(self.config)
        self.task_loader.load_dataset()
        self.task_loader.make_dataloaders()

        self.train_loader = self.task_loader.train_loader
        self.dev_loader = self.task_loader.dev_loader
        self.test_loader = self.task_loader.test_loader

    def load_model(self):
        self.task_loader.update_config_for_model(self.config['model'])
        self.model = build_model(self.config['model'])

    def load_optimizer(self):
        optimizer_class = getattr(optim, self.config['optimizer']['method'])

        parameters = self.model.parameters()

        if hasattr(self.model, 'get_optimizer_parameters'):
            parameters = self.model.get_optimizer_parameters(self.config)

        self.optimizer = optimizer_class(parameters,
                                         **self.config['optimizer']['params'])

    def load_extras(self):
        self.criterion = Loss(self.config['loss'])

        if self.config['lr_scheduler'] is True:
            self.lr_scheduler = optim.LambdaLR(self.optimizer,
                                               lr_lambda=lr_lambda_update)

    def config_based_setup(self):
        torch.manual_seed(self.config['seed'])

        if self.config['use_cuda']:
            torch.cuda.manual_seed(self.config['seed'])

    def train(self):
        training_parameters = self.config['training_parameters']
        log_interval = training_parameters['log_interval']
        snapshot_interval = training_parameters['snapshot_interval']
        max_iterations = training_parameters['max_iterations']
        should_clip_gradients = training_parameters['clip_gradients']
        max_epochs = self.config['max_epochs']

        should_check_on_epoch = False

        if max_epochs is not None:
            should_check_on_epoch = True
            max_iterations = math.inf

        current_epoch = 0
        current_iteration = 0

        self.model.train()

        while current_iteration < max_iterations:
            current_epoch += 1
            if should_check_on_epoch and current_epoch > max_epochs:
                break

            for batch in self.train_loader:
                current_iteration += 1

                if current_iteration > max_iterations:
                    break

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(current_iteration, self.config)

                self.optimizer.zero_grad()

                data, y = self.train_loader.prepare_data(batch)

                if not isinstance(data, list):
                    data = [data]

                output = self.model(*data)
                loss = self.criterion(output, y)

                loss.backward()

                if should_clip_gradients:
                    clip_gradients(self.model, current_iteration, self.writer,
                                   self.config)

                self.optimizer.step()

                # TODO: Add to meter here

                if current_iteration % log_interval:
                    # TODO: Implement logging
                    # TODO: Do validation check here
                    self.evaluate()
                    self.writer.save_log()

                if current_iteration % snapshot_interval:
                    # TODO: Implement checkpointing
                    self.checkpoint()

    def evaluate(self):
        return
