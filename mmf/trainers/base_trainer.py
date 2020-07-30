# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod

from mmf.common import typings as mmf_typings
from mmf.common.registry import registry


@registry.register_trainer("base")
class BaseTrainer(ABC):
    def __init__(self, config: mmf_typings.DictConfig):
        self.config = config
        self.training_config = self.config.training

    def load(self):
        # Set run type
        self.run_type = self.config.get("run_type", "train")

        # Print configuration
        configuration = registry.get("configuration", no_warning=True)
        if configuration:
            configuration.pretty_print()

        # Configure device and cudnn deterministic
        self.configure_device()
        self.configure_seed()

        # Load dataset, model, optimizer and metrics
        self.load_datasets()
        self.load_model()
        self.load_optimizer()
        self.load_metrics()

        # Initialize Callbacks
        self.configure_callbacks()

    @abstractmethod
    def configure_device(self):
        """Warning: this is just empty shell for code implemented in other class.
        Configure and set device properties here.
        """

    @abstractmethod
    def configure_seed(self):
        """Configure seed and related changes like torch deterministic etc shere.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def configure_callbacks(self):
        """Configure callbacks and add callbacks be executed during
        different events during training, validation or test.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def load_datasets(self):
        """Loads datasets and dataloaders.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def load_model(self):
        """Load the model.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def load_optimizer(self):
        """Load optimizers.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def load_metrics(self):
        """Load metrics for evaluation.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def train(self):
        """Runs full training and optimization.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def inference(self):
        """Runs inference and validation, generate predictions.

        Warning: Empty shell for code to be implemented in other class.
        """
