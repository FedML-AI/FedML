# Script for running distributed clients using torchrun

import sys
from fedml.arguments import load_arguments
import subprocess
import os
def launch_dist_trainers():
    inputs = sys.argv[1:]
    args = load_arguments()
    os.environ['PDSH_RCMD_TYPE'] = 'ssh'
    node_addresses = ",".join(args.node_addresses)
    pdsh_cmd_aruments = ['pdsh', '-w', node_addresses]
    torchrun_path = subprocess.run(['which', 'torchrun'], capture_output=True, text=True).stdout.strip()
    torchrun_cmd_arguments = [
        f"cd {os.path.abspath('.')};",
        torchrun_path,
        f"--nnodes={args.n_node_in_silo}",
        f"--nproc_per_node={args.n_proc_per_node}",
        f"--rdzv_endpoint={args.pg_master_address}:{args.pg_master_port}",
        "--node_rank=%n",
        "--rdzv_id=hi_fl",
        "torch_client.py",
    ] + inputs

    # print(" ".join(torchrun_arguments))

    subprocess.run(pdsh_cmd_aruments + torchrun_cmd_arguments)


