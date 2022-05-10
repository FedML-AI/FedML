import torch

import logging


def get_device(args):
    if args.training_type == "simulation" and args.backend == "single_process":
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
        if args.scenario == "horizontal":
            if args.using_gpu:
                device = torch.device(
                    "cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu"
                )
            else:
                device = torch.device("cpu")
            logging.info("device = {}".format(device))
        elif args.scenario == "hierarchical":
            from .gpu_mapping import (
                mapping_processes_to_gpu_device_from_yaml_file,
            )

            device = mapping_processes_to_gpu_device_from_yaml_file(
                args.rank_in_node,
                args.n_proc_per_node,
                args.gpu_mapping_file if args.using_gpu else None,
                args.gpu_mapping_key if args.using_gpu else None,
            )
            logging.info("device = {}".format(device))
            return device
        else:
            if args.using_gpu:
                device = torch.device(
                    "cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu"
                )
            else:
                device = torch.device("cpu")
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
