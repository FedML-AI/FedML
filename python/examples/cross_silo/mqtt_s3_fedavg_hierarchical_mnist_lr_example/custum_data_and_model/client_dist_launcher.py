# Script for running distributed clients using torchrun

import sys
from fedml.arguments import load_arguments
import subprocess

inputs = sys.argv[1:]
args = load_arguments()
nproc_per_node = str(args.n_proc_per_node)
torchrun_arguments = ["--standalone", "--nnodes=1", "--nproc_per_node="+ nproc_per_node , "torch_client.py"] + inputs
subprocess.run(["torchrun"] + torchrun_arguments)


# mpi_arguments = ["--np", nproc_per_node, "python3" , "torch_client.py"] + inputs
# subprocess.run(["mpirun"] + mpi_arguments)
