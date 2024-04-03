import logging
import socket

import yaml

from fedml.constants import FEDML_CROSS_CLOUD_SCENARIO_HIERARCHICAL, FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL
from ..ml.engine import ml_engine_adapter


def mapping_processes_to_gpu_device_from_yaml_file_cross_silo(
    process_id, worker_number, gpu_util_file, gpu_util_key, device_type, scenario, gpu_id=None, args=None
):
    if device_type != "gpu":
        args.using_gpu = False
        device = ml_engine_adapter.get_device(args, device_id=gpu_id, device_type=device_type)
        logging.info(f"Training on device: {device}")
        return device
    else:
        if gpu_id is not None:
            args.using_gpu = True
            device = ml_engine_adapter.get_device(args, device_id=gpu_id, device_type=device_type)
        elif gpu_util_file is None:
            args.using_gpu = True
            device = ml_engine_adapter.get_device(args, device_id=gpu_id, device_type=device_type)
        else:
            unique_gpu = scenario in (FEDML_CROSS_CLOUD_SCENARIO_HIERARCHICAL, FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL)

            with open(gpu_util_file, "r") as f:
                gpu_util_yaml = yaml.load(f, Loader=yaml.FullLoader)
                # gpu_util_num_process = 'gpu_util_' + str(worker_number)
                # gpu_util = gpu_util_yaml[gpu_util_num_process]
                gpu_util = gpu_util_yaml[gpu_util_key]
                logging.info("gpu_util = {}".format(gpu_util))
                gpu_util_map = {}
                i = 0
                for host, gpus_util_map_host in gpu_util.items():
                    for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
                        # validate DDP gpu mapping
                        if unique_gpu and num_process_on_gpu > 1:
                            raise Exception(
                                "Cannot put {num_process_on_gpu} processes on GPU {gpu_j} of {host}."
                                "PyTorch DDP supports up to one process on each GPU."
                            )
                        for _ in range(num_process_on_gpu):
                            gpu_util_map[i] = (host, gpu_j)
                            i += 1

                logging.info(
                    "Process %d running on host: %s, gethostname: %s, local_gpu_id: %d ..."
                    % (process_id, gpu_util_map[process_id][0], socket.gethostname(), gpu_util_map[process_id][1],)
                )
                logging.info("i = {}, worker_number = {}".format(i, worker_number))
                assert i == worker_number, f"Invalid GPU Number. Expected {worker_number}, Received {i}."

            args.using_gpu = True
            device = ml_engine_adapter.get_device(args, device_id=str(gpu_util_map[process_id][1]), device_type="gpu")

        logging.info("process_id = {}, GPU device = {}".format(process_id, device))
        return device
