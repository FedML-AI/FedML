from common import (
    identity,
    identity_script,
    heavy,
    heavy_script,
    identity_cuda,
    identity_script_cuda,
    heavy_cuda,
    heavy_script_cuda,
    stamp_time,
    compute_delay,
    NUM_RPC,
)

from torch.distributed import rpc
from functools import partial
from statistics import stdev

import torch
import time
import os
import threading

GPU_ID = 5


def measure(*, name=None, func=None, args=None, cuda=False, out_file=None):
    # warmup
    futs = []
    for _ in range(NUM_RPC):
        futs.append(rpc.rpc_async("server", func, args=args))

    torch.futures.wait_all(futs)
    torch.cuda.current_stream(GPU_ID).synchronize()

    # warmup done
    timestamps = {}
    states = {
        "lock": threading.Lock(),
        "future": torch.futures.Future(),
        "pending": NUM_RPC,
    }

    def mark_complete(index, cuda, fut):
        timestamps[index]["tok"] = stamp_time(cuda)
        with states["lock"]:
            states["pending"] -= 1
            if states["pending"] == 0:
                states["future"].set_result(0)

    start = time.time()
    futs = []
    for index in range(NUM_RPC):
        timestamps[index] = {}
        timestamps[index]["tik"] = stamp_time(cuda)
        fut = rpc.rpc_async("server", func, args=args)
        futs.append(fut)
        fut.add_done_callback(partial(mark_complete, index, cuda))

    torch.futures.wait_all(futs)
    states["future"].wait()
    # torch.cuda.current_stream(GPU_ID).synchronize()

    delays = []
    for index in range(len(timestamps)):
        delays.append(compute_delay(timestamps[index], cuda))

    end = time.time()

    mean = sum(delays) / len(delays)
    stdv = stdev(delays)
    total = end - start
    print(
        f"{name}_{'cuda' if cuda else 'cpu'}: mean = {mean}, stdev = {stdv}, total = {end - start}",
        flush=True,
    )
    if out_file:
        out_file.write(f"{name}, {mean}, {stdv}, {total}\n")
    return mean, stdv, total


def run(addr="localhost", port="29500"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    assert torch.cuda.device_count() == 1

    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=256, device_maps={"server": {GPU_ID: GPU_ID}}
    )
    rpc.init_rpc("client", rank=1, world_size=2, rpc_backend_options=options)

    for size in [100, 1000, 10000]:
        # for size in [100, 1000]:
        print(f"======= size = {size} =====")
        f = open(f"logs/single_pt_rpc_{size}.log", "w")
        tensor = torch.ones(size, size)

        # identity
        measure(
            name="identity",
            func=identity,
            args=(tensor,),
            cuda=False,
            out_file=f,
        )

        # identity
        measure(
            name="identity",
            func=identity,
            args=(tensor,),
            cuda=False,
            out_file=f,
        )

        # identity script
        measure(
            name="identity_script",
            func=identity_script,
            args=(tensor,),
            cuda=False,
            out_file=f,
        )

        # heavy
        measure(
            name="heavy",
            func=heavy,
            args=(tensor,),
            cuda=False,
            out_file=f,
        )

        # heavy script
        measure(
            name="heavy_script",
            func=heavy_script,
            args=(tensor,),
            cuda=False,
            out_file=f,
        )

        tensor = tensor.to(GPU_ID)
        torch.cuda.current_stream(GPU_ID).synchronize()
        # identity cuda
        measure(
            name="identity",
            func=identity_cuda,
            args=(tensor,),
            cuda=True,
            out_file=f,
        )

        # identity script cuda
        measure(
            name="identity_script",
            func=identity_script_cuda,
            args=(tensor,),
            cuda=True,
            out_file=f,
        )

        # heavy cuda
        measure(
            name="heavy",
            func=heavy_cuda,
            args=(tensor,),
            cuda=True,
            out_file=f,
        )

        # heavy script cuda
        measure(
            name="heavy_script",
            func=heavy_script_cuda,
            args=(tensor,),
            cuda=True,
            out_file=f,
        )

        f.close()

    rpc.shutdown()
