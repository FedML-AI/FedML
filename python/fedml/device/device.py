import logging

import torch

from fedml.device.gpu_mapping import mapping_processes_to_gpu_device


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
        from .gpu_mapping import (
            mapping_processes_to_gpu_device_from_yaml_file,
        )

        device = mapping_processes_to_gpu_device_from_yaml_file(
            args.process_id,
            args.worker_num,
            args.gpu_mapping_file if args.using_gpu else None,
            args.gpu_mapping_key,
        )
        return device
    elif args.training_type == "cross_silo":
        from .gpu_mapping import (
            mapping_processes_to_gpu_device_from_yaml_file,
        )
        if args.scenario == "hierarchical":
            device = mapping_processes_to_gpu_device_from_yaml_file(
                args.proc_rank_in_silo,
                args.n_proc_in_silo,
                args.gpu_mapping_file if args.using_gpu else None,
                args.gpu_mapping_key if args.using_gpu else None,
            )
        else:
            device_type = "gpu" if not hasattr(args, "device_type") else args.device_type
            device = mapping_processes_to_gpu_device(
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
