# Copyright (c) Facebook, Inc. and its affiliates.

import gc
import logging
import warnings
from abc import ABC
from typing import Any, Dict

import torch
from mmf.common.registry import registry
from mmf.common.report import Report
from mmf.common.sample import to_device
from mmf.utils.general import clip_gradients
from torch import Tensor


logger = logging.getLogger(__name__)


class TrainerTrainingLoopMixin(ABC):
    current_epoch: int = 0
    current_iteration: int = 0
    num_updates: int = 0

    def training_loop(self) -> None:
        self.max_updates = self._calculate_max_updates()
        torch.autograd.set_detect_anomaly(self.training_config.detect_anomaly)

        logger.info("Starting training...")
        self.model.train()
        self.run_training_epoch()
        self.after_training_loop()

    def after_training_loop(self) -> None:
        logger.info("Stepping into final validation check")
        # Only do when run_type has train as it shouldn't happen on validation and
        # inference runs. Inference will take care of this anyways. Also, don't run
        # if current iteration is divisble by snapshot interval as it will just
        # be a repeat
        if (
            "train" in self.run_type
            and self.num_updates % self.training_config.evaluation_interval != 0
        ):
            # Create a new meter for this case
            report, meter = self.evaluation_loop(self.val_loader)

            # Validation end callbacks
            self.on_validation_end(report=report, meter=meter)

    def run_training_epoch(self) -> None:
        should_break = False
        while self.num_updates < self.max_updates and not should_break:
            self.current_epoch += 1
            registry.register("current_epoch", self.current_epoch)

            # Seed the sampler in case if it is distributed
            self.dataset_loader.seed_sampler("train", self.current_epoch)

            for batch in self.train_loader:
                self.profile("Batch load time")
                self.current_iteration += 1
                logger.debug(self.num_updates + 1)

                self.run_training_batch(batch)

                # Check if training should be stopped
                should_break = False

                if self.num_updates % self.training_config.evaluation_interval == 0:
                    # Validation begin callbacks
                    self.on_validation_start()

                    logger.info("Evaluation time. Running on full validation set...")
                    # Validation and Early stopping
                    # Create a new meter for this case
                    report, meter = self.evaluation_loop(self.val_loader)

                    # Validation end callbacks
                    stop = self.early_stop_callback.on_validation_end(
                        report=report, meter=meter
                    )
                    self.on_validation_end(report=report, meter=meter)

                    gc.collect()

                    if "cuda" in str(self.device):
                        torch.cuda.empty_cache()

                    if stop is True:
                        logger.info("Early stopping activated")
                        should_break = True
                if self.num_updates >= self.max_updates:
                    should_break = True

                if should_break:
                    break

    def run_training_batch(self, batch: Tensor) -> None:
        # Train batch start callbacks
        self.on_batch_start()

        report = self._forward(batch)
        loss = self._extract_loss(report)
        self._backward(loss)

        should_log = False
        if self.num_updates % self.logistics_callback.log_interval == 0:
            should_log = True
            # Calculate metrics every log interval for debugging
            if self.training_config.evaluate_metrics:
                report.metrics = self.metrics(report, report)
            self.update_meter(report, self.meter)

        # Train batch end callbacks
        self.on_batch_end(report=report, meter=self.meter, should_log=should_log)

    def _forward(self, batch: Tensor) -> Dict[str, Any]:
        prepared_batch = self.dataset_loader.prepare_batch(batch)
        # Move the sample list to device if it isn't as of now.
        prepared_batch = to_device(prepared_batch, torch.device("cuda"))
        self.profile("Batch prepare time")
        # Arguments should be a dict at this point
        model_output = self.model(prepared_batch)
        report = Report(prepared_batch, model_output)
        self.profile("Forward time")

        return report

    def _backward(self, loss: Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()

        if self.training_config.clip_gradients:
            clip_gradients(
                self.model,
                self.num_updates,
                self.logistics_callback.tb_writer,
                self.config,
            )

        self.optimizer.step()
        self.num_updates += 1
        self.profile("Backward time")

    def _extract_loss(self, report: Dict[str, Any]) -> Tensor:
        loss_dict = report.losses
        loss = sum([loss.mean() for loss in loss_dict.values()])
        return loss

    def _calculate_max_updates(self):
        max_updates = self.training_config.max_updates
        max_epochs = self.training_config.max_epochs
        if max_updates is None and max_epochs is None:
            raise ValueError("Neither max_updates nor max_epochs is specified.")

        if max_updates is not None and max_epochs is not None:
            warnings.warn(
                "Both max_updates and max_epochs are specified. "
                + f"Favoring max_epochs: {max_epochs}"
            )

        if max_epochs is not None:
            max_updates = len(self.train_loader) * max_epochs

        return max_updates


class TrainerRetrieverTrainingLoopMixin(TrainerTrainingLoopMixin):
    def _forward(self, batch: Tensor) -> Dict[str, Any]:
        batch = self.dataset_loader.prepare_batch(batch)
        self.profile("Batch prepare time")

        retrieve_modal = self.config.rag_config.retrieve_mode
        if torch.distributed.is_initialized():
            model = self.model.module
        else:
            model = self.model

        model_output = {}
        retrieve_modal = self.config.rag_config.retrieve_mode

        if "image" in retrieve_modal:
            stack_images = torch.cat((batch.image, batch.pos_image), 0)
            image_embeds = model.ref_encode_image(stack_images)
            v = int(stack_images.shape[0] / 2)

            Q_image_embeds = model.convert_dims(image_embeds[:v])
            A_image_embeds = model.convert_dims(image_embeds[v:])

            sim_scores = model.calc_batch_sim(Q_image_embeds, A_image_embeds)

        if "text" in retrieve_modal:
            Q_text_embeds = model.convert_dims(model.ref_encode_text(
                batch.input_ids, batch.segment_ids)
            )
            A_text_embeds = model.convert_dims(model.ref_encode_text(
                batch.pos_input_ids, batch.pos_segment_ids)
            )

            sim_scores = model.calc_batch_sim(Q_text_embeds, A_text_embeds)

        if "image_text" == retrieve_modal:
            Q_concat = torch.cat((Q_image_embeds, Q_text_embeds), 1)
            A_concat = torch.cat((A_image_embeds, A_text_embeds), 1)
            sim_scores = model.calc_batch_sim(Q_concat, A_concat)

        bs = len(batch['id'])
        device = torch.cuda.current_device()
        batch['targets'] = torch.arange(0, bs, dtype=torch.long).to(device=device)

        model_output['scores'] = sim_scores
        model_output['losses'] = model.losses(batch, model_output)
        report = Report(batch, model_output)
        self.profile("Forward time")
        return report
