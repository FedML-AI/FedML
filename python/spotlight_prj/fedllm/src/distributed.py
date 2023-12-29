from typing import ContextManager, Iterable, Optional, Union

from contextlib import contextmanager, nullcontext

from fedml.train.llm.distributed import is_deepspeed_initialized, is_deepspeed_module
from fedml.train.llm.integrations import is_deepspeed_available, is_deepspeed_zero3_enabled
from torch import distributed as dist
from torch.nn import Module, Parameter

if is_deepspeed_available():
    import deepspeed
    from deepspeed.runtime.zero import GatheredParameters


def get_rank() -> int:
    if is_deepspeed_initialized():
        return deepspeed.comm.get_rank()
    elif dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


@contextmanager
def gather_parameter(
        params: Union[Iterable[Parameter], Parameter],
        modifier_rank: Optional[int] = None,
        fwd_module: Optional[Module] = None,
        enabled: bool = True
) -> ContextManager[None]:
    if enabled and is_deepspeed_initialized() and is_deepspeed_zero3_enabled() and is_deepspeed_module(params):
        context = GatheredParameters(params, modifier_rank, fwd_module, enabled)
    else:
        context = nullcontext()

    with context:
        yield
