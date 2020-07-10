# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC
from typing import Type

from mmf.utils.timer import Timer


class TrainerProfilingMixin(ABC):
    profiler: Type[Timer] = Timer()

    def profile(self, text: str) -> None:
        if self.training_config.logger_level != "debug":
            return
        self.writer.write(text + ": " + self.profiler.get_time_since_start(), "debug")
        self.profiler.reset()
