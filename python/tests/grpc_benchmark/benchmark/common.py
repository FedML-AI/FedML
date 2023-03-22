import torch
from torch import Tensor
from typing import Tuple
import time

# there are only 32 streams in the pool, keep this nunber below that.
NUM_RPC = 10


def identity(x: Tensor) -> Tensor:
    return x


@torch.jit.script
def identity_script(x: Tensor) -> Tensor:
    return x


# grpc + cuda requries sync=True
def heavy(x: Tensor) -> Tensor:
    for _ in range(100):
        x *= 2.0
        x /= 2.0
    return x


@torch.jit.script
def heavy_script(x: Tensor) -> Tensor:
    for _ in range(100):
        x *= 2.0
        x /= 2.0
    return x


def identity_cuda(x: Tensor) -> Tensor:
    return x.cuda(0)


@torch.jit.script
def identity_script_cuda(x: Tensor) -> Tensor:
    return x.to(0)


# grpc + cuda requries sync=True
def heavy_cuda(x: Tensor) -> Tensor:
    for _ in range(100):
        x *= 2.0
        x /= 2.0
    return x.cuda(0)


@torch.jit.script
def heavy_script_cuda(x: Tensor) -> Tensor:
    for _ in range(100):
        x *= 2.0
        x /= 2.0
    return x.to(0)


def stamp_time(cuda=False):
    if cuda:
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(0))
        return event
    else:
        return time.time()


def compute_delay(ts, cuda=False):
    if cuda:
        ts["tok"].synchronize()
        return ts["tik"].elapsed_time(ts["tok"]) / 1e3
    else:
        return ts["tok"] - ts["tik"]
