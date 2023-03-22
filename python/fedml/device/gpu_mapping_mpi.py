import logging
import socket

import yaml

from ..ml.engine import ml_engine_adapter


def mapping_processes_to_gpu_device_from_yaml_file_mpi(
    process_id, worker_number, gpu_util_file, gpu_util_key, args=None
):
    if gpu_util_file is None:
        logging.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(" ################## You do not indicate gpu_util_file, will use CPU training  #################")
        device = ml_engine_adapter.get_device(args, device_type="cpu")
        logging.info(device)
        return device
    else:
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
                    for _ in range(num_process_on_gpu):
                        gpu_util_map[i] = (host, gpu_j)
                        i += 1
            logging.info(
                "Process %d running on host: %s, gethostname: %s, local_gpu_id: %d ..."
                % (process_id, gpu_util_map[process_id][0], socket.gethostname(), gpu_util_map[process_id][1],)
            )
            logging.info("i = {}, worker_number = {}".format(i, worker_number))
            assert i == worker_number

        args.gpu_id = gpu_util_map[process_id][1]
        device = ml_engine_adapter.get_device(args, device_id=str(gpu_util_map[process_id][1]), device_type="gpu")
        logging.info("process_id = {}, GPU device = {}".format(process_id, device))
        return device


def mapping_processes_to_gpu_device_from_gpu_util_parse(process_id, worker_number, gpu_util_parse, args=None):
    if gpu_util_parse == None:
        device = ml_engine_adapter.get_device(args, device_type="cpu")
        logging.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(" ##################  Not Indicate gpu_util_file, using cpu  #################")
        logging.info(device)
        # return gpu_util_map[process_id][1]
        return device
    else:
        # example parse str `gpu_util_parse`:
        # "gpu1:0,1,1,2;gpu2:3,3,3;gpu3:0,0,0,1,2,4,4,0"
        gpu_util_parse_temp = gpu_util_parse.split(";")
        gpu_util_parse_temp = [(item.split(":")[0], item.split(":")[1]) for item in gpu_util_parse_temp]

        gpu_util = {}
        for (host, gpus_str) in gpu_util_parse_temp:
            gpu_util[host] = [int(num_process_on_gpu) for num_process_on_gpu in gpus_str.split(",")]

        gpu_util_map = {}
        i = 0
        for host, gpus_util_map_host in gpu_util.items():
            for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
                for _ in range(num_process_on_gpu):
                    gpu_util_map[i] = (host, gpu_j)
                    i += 1
        logging.info(
            "Process %d running on host: %s,gethostname: %s, gpu: %d ..."
            % (process_id, gpu_util_map[process_id][0], socket.gethostname(), gpu_util_map[process_id][1])
        )
        assert i == worker_number

        args.using_gpu = True
        args.gpu_id = gpu_util_map[process_id][1]
        device = ml_engine_adapter.get_device(args, device_id=gpu_util_map[process_id][1], device_type="gpu")
        logging.info(device)
        return device
