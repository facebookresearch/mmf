from typing import Union, List, Tuple, Optional

from PIL.Image import Image
from torch import Tensor

from mmf.utils.metric_definitions import Metric

MetricType = Union[float, int, Tensor, Image]
MetricResult = Optional[List['MetricValue']]


class AlternativeValues:
    """
    A container for alternative representations of the same metric value.
    """
    def __init__(self, *alternatives: MetricType):
        self.alternatives: Tuple[MetricType] = alternatives

    def best_supported_value(self, *supported_types: type) -> \
            Optional[MetricType]:
        """
        Retrieves a supported representation for this metric value.
        :param supported_types: A list of supported value types.
        :return: The best supported representation. Returns None if no supported
            representation is found.
        """
        for alternative in self.alternatives:
            if isinstance(alternative, supported_types):
                return alternative
        return None


class MetricValue(object):
    """
    The result of a Metric.
    A result has a name, a value and a "x" position in which the metric value
    should be plotted.
    The "value" field can also be an instance of "AlternativeValues", in which
    case it means that alternative representations exist for this value. For
    instance, the Confusion Matrix can be represented both as a Tensor and as
    an Image. It's up to the Logger, according to its capabilities, decide which
    representation to use.
    """
    def __init__(self, origin: Metric, name: str,
                 value: Union[MetricType, AlternativeValues], x_plot: int):
        """
        Creates an instance of MetricValue.
        :param origin: The originating Metric instance.
        :param name: The display name of this value. This value roughly
            corresponds to the name of the plot in which the value should
            be logged.
        :param value: The value of the metric. Can be a scalar value,
            a PIL Image, or a Tensor. If more than a possible representation
            of the same value exist, an instance of :class:`AlternativeValues`
            can be passed. For instance, the Confusion Matrix can be represented
            both as an Image and a Tensor, in which case an instance of
            :class:`AlternativeValues` carrying both the Tensor and the Image
            is more appropriate. The Logger instance will then select the most
            appropriate way to log the metric according to its capabilities.
        :param x_plot: The position of the value. This value roughly corresponds
            to the x-axis position of the value in a plot. When logging a
            singleton value, pass 0 as a value for this parameter.
        """
        self.origin: Metric = origin
        self.name: str = name
        self.value: Union[MetricType, AlternativeValues] = value
        self.x_plot: int = x_plot


__all__ = [
    'MetricType',
    'MetricResult',
    'AlternativeValues',
    'MetricValue'
]