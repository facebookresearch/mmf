# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod

from mmf.common import typings as mmf_typings
from mmf.common.registry import registry

<<<<<<< HEAD
=======
LXMERT_AUG = True
TINY = True

if TINY:
    SPLITS = ["mscoco_minival"]
else:
    SPLITS = ["mscoco_train","mscoco_nominival","vgnococo"]

if LXMERT_AUG:
    import sys
    sys.path.append("/playpen/lxmert_loader")
    from lxmert_pretrain import get_tuple
>>>>>>> finalize configs + more temp model changes for new input data

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

<<<<<<< HEAD
        Warning: Empty shell for code to be implemented in other class.
        """
=======
            # try with tiny first
            train_tuple = get_tuple(SPLITS,
                    256,
                    shuffle=True,
                    drop_last=True)
            valid_batch_size = 512
            valid_tuple = get_tuple(
                    ["mscoco_minival"],
                    valid_batch_size,
                    shuffle=False,
                    drop_last=False,
                    topk=5000)
>>>>>>> finalize configs + more temp model changes for new input data

    @abstractmethod
    def load_optimizer(self):
        """Load optimizers.

<<<<<<< HEAD
        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
=======
>>>>>>> finalize configs + more temp model changes for new input data
    def load_metrics(self):
        """Load metrics for evaluation.

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def train(self):
<<<<<<< HEAD
        """Runs full training and optimization.
=======
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
>>>>>>> finalize configs + more temp model changes for new input data

        Warning: Empty shell for code to be implemented in other class.
        """

    @abstractmethod
    def inference(self):
        """Runs inference and validation, generate predictions.

        Warning: Empty shell for code to be implemented in other class.
        """
