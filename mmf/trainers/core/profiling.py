# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import threading
from abc import ABC
from typing import Type

from mmf.utils.timer import Timer


logger = logging.getLogger(__name__)


class TrainerProfilingMixin(ABC):
    profiler: Type[Timer] = Timer()

    def profile(self, text: str) -> None:
        if self.training_config.logger_level != "debug":
            return
        logging.debug(
            f"tid={threading.current_thread().ident}, {text}: {self.profiler.get_time_since_start()}"
        )
        self.profiler.reset()
