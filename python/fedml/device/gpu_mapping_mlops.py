import logging
import socket

import torch
import yaml


def mapping_processes_to_gpu_device_from_yaml_file(
    process_id, worker_number, gpu_util_file, gpu_util_key, check_cross_silo=False
):
    if gpu_util_file == None:
        device = torch.device("cpu")
        logging.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(
            " ################## You do not indicate gpu_util_file, will use CPU training  #################"
        )
        logging.info(device)
        # return gpu_util_map[process_id][1]
        return device
    else:
        (gpu_util_map, mapping_num) = get_gpu_util_map(
            gpu_util_file, gpu_util_key, check_cross_silo
        )
        logging.info(
            "Process %d running on host: %s, gethostname: %s, local_gpu_id: %d ..."
            % (
                process_id,
                gpu_util_map[process_id][0],
                socket.gethostname(),
                gpu_util_map[process_id][1],
            )
        )
        logging.info("i = {}, worker_number = {}".format(mapping_num, worker_number))
        assert mapping_num == worker_number
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_util_map[process_id][1])
        device = torch.device(
            "cuda:" + str(gpu_util_map[process_id][1])
            if torch.cuda.is_available()
            else "cpu"
        )
        logging.info("process_id = {}, GPU device = {}".format(process_id, device))
        # return gpu_util_map[process_id][1]
        return device


def get_gpu_util_map(gpu_util_file, gpu_util_key, check_cross_silo=False):
    with open(gpu_util_file, "r") as f:
        gpu_util_yaml = yaml.load(f, Loader=yaml.FullLoader)
        # gpu_util_num_process = 'gpu_util_' + str(worker_number)
        # gpu_util = gpu_util_yaml[gpu_util_num_process]
        gpu_util = gpu_util_yaml[gpu_util_key]
        logging.info("gpu_util = {}".format(gpu_util))
        gpu_util_map = {}
        i = 0
        for host, gpus_util_map_host in gpu_util.items():
            gpu_found = False
            for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
                for _ in range(num_process_on_gpu):
                    if gpu_found and check_cross_silo:
                        raise Exception(
                            "Only one gpu should be specifed for silo server"
                        )
                    gpu_util_map[i] = (host, gpu_j)
                    gpu_found = True
                    i += 1
    return (gpu_util_map, i)
