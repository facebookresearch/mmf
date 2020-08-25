# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common import typings as mmf_typings
from mmf.common.registry import registry
from mmf.trainers.mmf_trainer import MMFTrainer
from mmf.trainers.core.training_loop import TrainerRetrieverTrainingLoopMixin


@registry.register_trainer("mmf_ret")
class MMFRetTrainer(
    MMFTrainer,
    TrainerRetrieverTrainingLoopMixin,
):
    def __init__(self, config: mmf_typings.DictConfig):
        super().__init__(config)
        TrainerRetrieverTrainingLoopMixin.__init__(config)
