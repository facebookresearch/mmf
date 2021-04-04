from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Dict, TYPE_CHECKING

from typing_extensions import Protocol

TResult = TypeVar('TResult')


class Metric(Protocol[TResult]):
    """
    Definition of a standalone metric.
    A standalone metric exposes methods to reset its internal state and
    to emit a result. Emitting a result does not automatically cause
    a reset in the internal state.
    The specific metric implementation exposes ways to update the internal
    state. Usually, standalone metrics like :class:`Sum`, :class:`Mean`,
    :class:`Accuracy`, ... expose an `update` method.
    The `Metric` class can be used as a standalone metric by directly calling
    its methods.
    """

    def result(self) -> Optional[TResult]:
        """
        Obtains the value of the metric.
        :return: The value of the metric.
        """
        pass

    def reset(self) -> None:
        """
        Resets the metric internal state.
        :return: None.
        """
        pass