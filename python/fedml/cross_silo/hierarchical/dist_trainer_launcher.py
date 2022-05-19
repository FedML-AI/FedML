# Script for running distributed clients using torchrun

import sys
from fedml.arguments import load_arguments
import subprocess
import os
from fedml.constants import FEDML_TRAINING_PLATFORM_CROSS_SILO

# env_variables = {
#     'NCCL_DEBUG':'INFO',
#     'NCCL_MIN_NRINGS':1,
#     'NCCL_TREE_THRESHOLD':4294967296,
#     'OMP_NUM_THREADS':8,
#     'NCCL_NSOCKS_PERTHREAD':8,
#     'NCCL_SOCKET_NTHREADS':8,
#     'NCCL_BUFFSIZE':1048576,
#     'NCCL_IB_DISABLE'=1
#     'NCCL_SOCKET_IFNAME'='$NETWORK_INTERFACE'
#     'GLOO_SOCKET_IFNAME'=$'NETWORK_INTERFACE'
#     'TP_SOCKET_IFNAME'=$'NETWORK_INTERFACE'
# }

def launch_dist_trainers():
    inputs = sys.argv[1:]
    args = load_arguments(FEDML_TRAINING_PLATFORM_CROSS_SILO)
    os.environ['PDSH_RCMD_TYPE'] = 'ssh'
    node_addresses = ",".join(args.node_addresses)
    pdsh_cmd_aruments = ['pdsh', '-w', node_addresses]
    torchrun_path = subprocess.run(['which', 'torchrun'], capture_output=True, text=True).stdout.strip()

    # exports = ""
    # for key, val in self.exports.items():
    #     exports += "export {}={}; ".format(key, val)

    torchrun_cmd_arguments = [
        # exports,
        f"cd {os.path.abspath('.')};",
        torchrun_path,
        f"--nnodes={args.n_node_in_silo}",
        f"--nproc_per_node={args.n_proc_per_node}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={args.master_address}:{args.launcher_rdzv_port}",
        "--node_rank=%n",
        "--rdzv_id=hi_fl",
        "torch_client.py",
    ] + inputs

    subprocess.run(pdsh_cmd_aruments + torchrun_cmd_arguments)

