from typing import Iterable, Union

from torch import distributed as dist
from torch.nn import Module, Parameter

from .integrations import is_deepspeed_available

if is_deepspeed_available():
    import deepspeed.comm

    is_deepspeed_initialized = deepspeed.comm.is_initialized
else:
    def is_deepspeed_initialized() -> bool:
        return False


def barrier() -> None:
    if is_deepspeed_initialized():
        deepspeed.comm.barrier()
    elif dist.is_initialized():
        dist.barrier()


def is_deepspeed_module(module: Union[Module, Parameter, Iterable[Parameter]]) -> bool:
    if is_deepspeed_available():
        if isinstance(module, Module):
            return any(hasattr(p, "ds_id") for p in module.parameters())

        elif isinstance(module, Parameter):
            return hasattr(module, "ds_id")

        elif isinstance(module, Iterable):
            return any(hasattr(p, "ds_id") for p in module if isinstance(p, Parameter))

    else:
        return False
