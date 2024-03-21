from typing import ContextManager, Iterable, Optional, Union

import contextlib
from contextlib import nullcontext

from torch import distributed as dist, Tensor
from torch.nn import Module, Parameter

from .integrations import is_deepspeed_available, is_deepspeed_zero3_enabled

if is_deepspeed_available():
    import deepspeed.comm
    from deepspeed.runtime.zero import GatheredParameters

    is_deepspeed_initialized = deepspeed.comm.is_initialized
else:
    def is_deepspeed_initialized() -> bool:
        return False


def barrier() -> None:
    if is_deepspeed_initialized():
        deepspeed.comm.barrier()
    elif dist.is_initialized():
        dist.barrier()


def get_rank() -> int:
    if is_deepspeed_initialized():
        return deepspeed.comm.get_rank()
    elif dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


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


# TODO: replace with `@contextmanager` once https://youtrack.jetbrains.com/issue/PY-36444 is fixed
@contextlib.contextmanager
def gather_parameter(
        params: Union[Iterable[Parameter], Parameter],
        modifier_rank: Optional[int] = None,
        fwd_module: Optional[Module] = None,
        enabled: bool = True
) -> ContextManager[None]:
    if isinstance(params, Iterable) and not isinstance(params, Tensor):
        # handle generator such as `model.parameters()`
        params = list(params)

    if enabled and is_deepspeed_initialized() and is_deepspeed_zero3_enabled():
        context = GatheredParameters(params, modifier_rank, fwd_module, enabled)
    else:
        context = nullcontext()

    with context:
        yield
