#!/usr/bin/env python3
"""Custom launcher for FedLLM with full model training support.

This launcher points to run_fedllm_custom.py instead of run_fedllm.py to use
the fixed trainer/aggregator classes that properly handle peft_type="none".
"""

import os
import sys
from pathlib import Path

# Get the directory of the original launch script
fedllm_dir = Path(__file__).parent
sys.path.insert(0, str(fedllm_dir))

# Import everything from the original launch script
from launch_fedllm import *
import subprocess
from subprocess import CalledProcessError

# Override the main function to use our custom runner
def main() -> int:
    # go to project root directory
    os.chdir(Path(__file__).parent)

    # update environment variables
    # Note: Wandb mode is controlled by config file settings
    # if len(os.getenv("WANDB_MODE", "")) == 0:
    #     os.environ["WANDB_MODE"] = "disabled"

    # parse args
    args = parse_args()

    print(
        f"master_addr: {args.master_addr},"
        f" master_port: {args.master_port},"
        f" num_nodes: {args.num_nodes},"
        f" num_gpus: {args.num_gpus}"
    )

    cmd = []
    if args.launcher == "deepspeed":
        cmd.extend([
            f"-m",
            f"deepspeed.launcher.runner",
            f"--master_addr", f"{args.master_addr}",
            f"--master_port", f"{args.master_port}",
        ])

        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            # when `CUDA_VISIBLE_DEVICES` is not specified, use all GPUs by setting `--num_nodes`
            cmd.extend([
                f"--num_nodes", f"{args.num_nodes}",
                f"--num_gpus", f"{args.num_gpus}",
            ])
        else:
            # see https://github.com/microsoft/DeepSpeed/issues/662
            # use `--include` to select GPUs and unset `CUDA_VISIBLE_DEVICES`
            cmd.extend([
                f"--include", f"{args.master_addr}:{os.getenv('CUDA_VISIBLE_DEVICES')}",
            ])
            os.environ.pop("CUDA_VISIBLE_DEVICES")

    elif args.launcher == "torch":
        cmd.extend([
            f"-m",
            f"torch.distributed.run",
            f"--nnodes", f"{args.num_nodes}",
            f"--nproc_per_node", f"{args.num_gpus}",
            f"--rdzv_endpoint", f"{args.master_addr}:{args.master_port}",
            f"--rdzv_backend", f"c10d"
        ])

    elif args.launcher != "python":
        raise ValueError(f"Unsupported launcher type \"{args.launcher}\".")

    cmd.extend([
        "run_fedllm_custom.py",  # USE CUSTOM RUNNER HERE
    ])

    print(f"cmd = {cmd}")
    print(f"sys.argv = {sys.argv}")

    proc = subprocess.run(
        [
            sys.executable,
            *cmd,
            *sys.argv[1:],
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ
    )

    print(f"{__file__} done.")
    if proc.returncode != 0:
        raise CalledProcessError(proc.returncode, proc.args)

    return proc.returncode


if __name__ == "__main__":
    exit(main()) 