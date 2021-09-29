from packaging import version
import torch
from typing import Any, Iterator, Tuple
from torch.nn import Parameter
from fairscale import __version__ as fairscale_version


if version.parse(fairscale_version) >= version.parse("0.3"):
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
    has_FSDP = True
else:
    FSDP = torch.nn.Module
    has_FSDP = False



class FullyShardedDataParallel(FSDP):
    def __init__(self, *args: Any, **kwargs: Any):
        if not has_FSDP:
            raise ImportError(
                "FairScale version >= 0.3 required for using training.fsdp.enabled=True"
            )
        super().__init__(*args, **kwargs)

    def all_named_parameters(self, *args: Any, **kwargs: Any) -> Iterator[Tuple[str, Parameter]]:
        with self.summon_full_params():
            return super().named_parameters(*args, **kwargs)

    def all_parameters(self, *args: Any, **kwargs: Any) -> Iterator[Parameter]:
        with self.summon_full_params():
            return super().parameters(*args, **kwargs)
