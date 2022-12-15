import logging

from ..ml.engine import ml_engine_adapter


from fedml.constants import (
    FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL,
    FEDML_TRAINING_PLATFORM_CROSS_SILO,
)


def get_device_type(args):
    if hasattr(args, "device_type"):
        if args.device_type == "cpu":
            device_type = "cpu"
        elif args.device_type == "gpu":
            if ml_engine_adapter.is_device_available(args, args.device_type):
                device_type = "gpu"
            else:
                print("ML engine install was not built with GPU enabled")
                device_type = "cpu"
        elif args.device_type == "mps":
            # Macbook M1: https://pytorch.org/docs/master/notes/mps.html
            if not ml_engine_adapter.is_device_available(args, args.device_type):
                device_type = "cpu"
            else:
                device_type = "mps"
        else:
            raise Exception("do not support device type = {}".format(args.device_type))
    else:
        if args.using_gpu:
            if ml_engine_adapter.is_device_available(args, "gpu"):
                device_type = "gpu"
            else:
                print("ML engine install was not built with GPU enabled")
                device_type = "cpu"
        else:
            device_type = "cpu"
    return device_type


def get_device(args):
    if args.training_type == "simulation" and args.backend == "sp":
        if not hasattr(args, "gpu_id"):
            args.gpu_id = 0
        device_type = get_device_type(args)
        device = ml_engine_adapter.get_device(args, args.gpu_id, device_type)
        logging.info("device = {}".format(device))
        return device
    elif args.training_type == "simulation" and args.backend == "MPI":
        from .gpu_mapping_mpi import (
            mapping_processes_to_gpu_device_from_yaml_file_mpi,
            mapping_processes_to_gpu_device_from_gpu_util_parse,
        )

        if hasattr(args, "gpu_util_parse"):
            device = mapping_processes_to_gpu_device_from_gpu_util_parse(
                args.process_id, args.worker_num, args.gpu_util_parse, args=args
            )
        else:
            device = mapping_processes_to_gpu_device_from_yaml_file_mpi(
                args.process_id,
                args.worker_num,
                args.gpu_mapping_file if args.using_gpu else None,
                args.gpu_mapping_key,
                args=args
            )
        logging.info("device = {}".format(device))
        return device
    elif args.training_type == "simulation" and args.backend == "NCCL":
        from .gpu_mapping_mpi import (
            mapping_processes_to_gpu_device_from_yaml_file_mpi,
            mapping_processes_to_gpu_device_from_gpu_util_parse,
        )

        device = mapping_processes_to_gpu_device_from_yaml_file_mpi(
            args.process_id,
            args.worker_num,
            args.gpu_mapping_file if args.using_gpu else None,
            args.gpu_mapping_key,
            args=args
        )
        logging.info("device = {}".format(device))
        return device
    elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO:

        from .gpu_mapping_cross_silo import (
            mapping_processes_to_gpu_device_from_yaml_file_cross_silo,
        )

        device_type = get_device_type(args)

        if args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            worker_number = args.n_proc_in_silo
            process_id = args.proc_rank_in_silo
        else:
            worker_number = args.worker_num + 1
            process_id = args.process_id

        if args.using_gpu:
            gpu_mapping_file = (
                args.gpu_mapping_file if hasattr(args, "gpu_mapping_file") else None
            )
            gpu_mapping_key = (
                args.gpu_mapping_key if hasattr(args, "gpu_mapping_key") else None
            )
            gpu_id = args.gpu_id if hasattr(args, "gpu_id") else None  # no no need to set gpu_id
        else:
            gpu_mapping_file = None
            gpu_mapping_key = None
            gpu_id = None

        logging.info(
            "devide_type = {}, gpu_mapping_file = {}, "
            "gpu_mapping_key = {}, gpu_id = {}".format(
                device_type, gpu_mapping_file, gpu_mapping_key, gpu_id
            )
        )
        scenario = args.scenario
        device = mapping_processes_to_gpu_device_from_yaml_file_cross_silo(
            process_id,
            worker_number,
            gpu_mapping_file,
            gpu_mapping_key,
            device_type,
            scenario,
            gpu_id,
            args=args
        )

        logging.info("device = {}".format(device))

        # Flag indicating the process will communicate with other clients
        is_master_process = (args.scenario == "horizontal") or (
            args.scenario == "hierarchical" and args.proc_rank_in_silo == 0
        )

        if args.enable_cuda_rpc and is_master_process:
            assert (
                device.index == args.cuda_rpc_gpu_mapping[args.rank]
            ), f"GPU assignemnt inconsistent with cuda_rpc_gpu_mapping. Assigned to GPU {device.index} while expecting {args.cuda_rpc_gpu_mapping[args.rank]}"

        return device
    elif args.training_type == "cross_device":
        device_type = get_device_type(args)
        device = ml_engine_adapter.get_device(args, device_type=device_type)
        logging.info("device = {}".format(device))
        return device
    else:
        raise Exception(
            "the training type {} is not defined!".format(args.training_type)
        )
