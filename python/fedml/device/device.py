import logging

import torch


def get_device(args):
    if args.training_type == "simulation" and args.backend == "sp":
        if args.using_gpu:
            device = torch.device(
                "cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu"
            )
        else:
            device = torch.device("cpu")
        logging.info("device = {}".format(device))
        return device
    elif args.training_type == "simulation" and args.backend == "MPI":
        from .gpu_mapping_mpi import (
            mapping_processes_to_gpu_device_from_yaml_file_mpi,
            mapping_processes_to_gpu_device_from_gpu_util_parse
        )
        if hasattr(args, "gpu_util_parse"):
            device = mapping_processes_to_gpu_device_from_gpu_util_parse(
                args.process_id,
                args.worker_num,
                args.gpu_util_parse,
            )
        else:
            device = mapping_processes_to_gpu_device_from_yaml_file_mpi(
                args.process_id,
                args.worker_num,
                args.gpu_mapping_file if args.using_gpu else None,
                args.gpu_mapping_key,
            )
        return device
    elif args.training_type == "simulation" and args.backend == "NCCL":
        from .gpu_mapping_mpi import (
            mapping_processes_to_gpu_device_from_yaml_file_mpi,
            mapping_processes_to_gpu_device_from_gpu_util_parse
        )

        device = mapping_processes_to_gpu_device_from_yaml_file_mpi(
            args.process_id,
            args.worker_num,
            args.gpu_mapping_file if args.using_gpu else None,
            args.gpu_mapping_key,
        )
        return device
    elif args.training_type == "cross_silo":

        if args.scenario == "hierarchical":
            from .gpu_mapping_cross_silo import (
                mapping_processes_to_gpu_device_from_yaml_file_cross_silo,
            )

            device = mapping_processes_to_gpu_device_from_yaml_file_cross_silo(
                args.proc_rank_in_silo,
                args.n_proc_in_silo,
                args.gpu_mapping_file if args.using_gpu else None,
                args.gpu_mapping_key if args.using_gpu else None,
            )
        else:
            from .gpu_mapping_cross_silo import (
                mapping_single_process_to_gpu_device_cross_silo,
            )

            device_type = (
                "gpu" if not hasattr(args, "device_type") else args.device_type
            )
            device = mapping_single_process_to_gpu_device_cross_silo(
                args.using_gpu, device_type
            )
        logging.info("device = {}".format(device))
        return device
    elif args.training_type == "cross_device":
        if args.using_gpu:
            device = torch.device(
                "cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu"
            )
        else:
            device = torch.device("cpu")
        logging.info("device = {}".format(device))
        return device
    else:
        raise Exception(
            "the training type {} is not defined!".format(args.training_type)
        )
