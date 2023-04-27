import os
import subprocess
import torch
from fedml.arguments import load_arguments
from fedml.constants import (
    FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL,
    FEDML_TRAINING_PLATFORM_CROSS_SILO,
    FEDML_CROSS_SILO_SCENARIO_HORIZONTAL,
)
from fedml.device import get_device_type

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


class CrossSiloLauncher:
    @staticmethod
    def launch_dist_trainers(torch_client_filename, inputs):
        # this is only used by the client (DDP or single process), so there is no need to specify the backend.
        args = load_arguments(FEDML_TRAINING_PLATFORM_CROSS_SILO)
        if args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            CrossSiloLauncher._run_cross_silo_hierarchical(args, torch_client_filename, inputs)
        elif args.scenario == FEDML_CROSS_SILO_SCENARIO_HORIZONTAL:
            CrossSiloLauncher._run_cross_silo_horizontal(args, torch_client_filename, inputs)
        else:
            raise Exception("we do not support {}, check whether this is typo in args.scenario".format(args.scenario))

    @staticmethod
    def _run_cross_silo_horizontal(args, torch_client_filename, inputs):
        python_path = subprocess.run(["which", "python"], capture_output=True, text=True).stdout.strip()
        process_arguments = [python_path, torch_client_filename] + inputs
        subprocess.run(process_arguments)

    @staticmethod
    def _run_cross_silo_hierarchical(args, torch_client_filename, inputs):
        def get_torchrun_arguments(node_rank):
            torchrun_path = subprocess.run(["which", "torchrun"], capture_output=True, text=True).stdout.strip()

            return [
                torchrun_path,
                f"--nnodes={args.n_node_in_silo}",
                f"--nproc_per_node={args.n_proc_per_node}",
                # "--rdzv_backend=c10d",
                f"--rdzv_endpoint={args.master_address}:{args.launcher_rdzv_port}",
                f"--node_rank={node_rank}",
                "--rdzv_id=hi_fl",
                torch_client_filename,
            ] + inputs

        network_interface = None if not hasattr(args, "network_interface") else args.network_interface
        print(f"Using network interface {network_interface} for process group and TRPC communication")
        env_variables = {
            "OMP_NUM_THREADS": "4",
        }
        if network_interface:
            env_variables = {
                **env_variables,
                "NCCL_SOCKET_IFNAME": network_interface,
                "GLOO_SOCKET_IFNAME": network_interface,
            }

        if args.n_node_in_silo == 1:
            args.node_rank = 0
            args.manual_launch = True
            if not (hasattr(args, "n_proc_per_node") and args.n_proc_per_node):
                print("Number of processes per node not specified.")
                device_type = get_device_type(args)
                if torch.cuda.is_available() and device_type == "gpu":
                    gpu_count = torch.cuda.device_count()
                    print(f"Using number of GPUs ({gpu_count}) as number of processeses.")
                    args.n_proc_per_node = gpu_count
                else:
                    print(f"Using number 1 as number of processeses.")
                    args.n_proc_per_node = 1

        if hasattr(args, "manual_launch") and args.manual_launch:
            print(f"Manual Client Launcher")
            node_rank = args.node_rank
            torchrun_cmd_arguments = get_torchrun_arguments(node_rank)
            process_args = torchrun_cmd_arguments
            print(f"Launching node {node_rank} of silo {args.rank}")
            subprocess.run(process_args, env=dict(os.environ, **env_variables))

        else:
            print(f"Automatic Client Launcher")

            which_pdsh = subprocess.run(["which", "pdsh"], capture_output=True, text=True).stdout.strip()

            if not which_pdsh:
                raise Exception(
                    f"Silo {args.rank} has {args.n_node_in_silo} nodes. Automatic Client Launcher for more than 1 nodes requires PSDH."
                )

            print(f"Launching nodes using pdsh")

            os.environ["PDSH_RCMD_TYPE"] = "ssh"
            node_addresses = ",".join(args.node_addresses)
            pdsh_cmd_aruments = ["pdsh", "-w", node_addresses]

            exports = ""
            for key, val in env_variables.items():
                exports += "export {}={}; ".format(key, val)
            prerun_args = [
                exports,
                f"cd {os.path.abspath('.')};",
            ]

            node_rank = "%n"
            torchrun_cmd_arguments = get_torchrun_arguments(node_rank)
            process_args = pdsh_cmd_aruments + prerun_args + torchrun_cmd_arguments
            subprocess.run(process_args)
