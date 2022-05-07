# Script for running distributed clients using torchrun

import sys
from fedml.arguments import load_arguments
import subprocess

def launch_dist_trainers():
    inputs = sys.argv[1:]
    args = load_arguments()
    torchrun_arguments = [
        f"--nnodes={args.n_node_in_silo}",
        f"--nproc_per_node={args.n_proc_per_node}",
        f"--rdzv_endpoint={args.pg_master_address}:{args.pg_master_port}",
        f"--node_rank={args.node_rank}",
        "--rdzv_id=hi_fl",
        "torch_client.py",
    ] + inputs

    print(" ".join(torchrun_arguments))

    subprocess.run(["torchrun"] + torchrun_arguments)


