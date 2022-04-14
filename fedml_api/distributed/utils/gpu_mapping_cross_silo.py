import logging
import socket

import torch
import yaml 


def corss_silo_mapping_processes_to_gpu_device_from_yaml_file(silo_rank, silo_proc_rank, silo_proc_num,client_silo_num_in_total, silo_gpu_util_file):
    if silo_gpu_util_file == None:
        device = torch.device("cpu")
        logging.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(" ################## You do not indicate silo_gpu_util_file, will use CPU training  #################")
        logging.info(device)
        # return gpu_util_map[process_id][1]
        return device
    else:
        gpu_util_map = get_gpu_util_map(silo_gpu_util_file)

        logging.info("Silo: %d, Silo proccess rank  %d running on host: %s, gethostname: %s,local_gpu_id: %d ..." % (silo_rank, silo_proc_rank, gpu_util_map[silo_proc_rank][0], socket.gethostname(), gpu_util_map[silo_proc_rank][1]))
        logging.info("i = {}, Number of silos = {}".format(len(gpu_util_map.items()), client_silo_num_in_total))
    
        logging.info("i = {}, Number of processes in silo = {}".format(len(gpu_util_map.items()), silo_proc_num))

        assert len(gpu_util_map.items()) == silo_proc_num, "Number of GPUs in gpu_mapping different from number of processes in the silo"

        
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_util_map[silo_proc_rank][1])
        device = torch.device("cuda:" + str(gpu_util_map[silo_proc_rank][1]) if torch.cuda.is_available() else "cpu")
        logging.info("process_id = {}, GPU device = {}".format(gpu_util_map[silo_proc_rank][1], device))
        # return gpu_util_map[process_id][1]
        return device


def get_gpu_util_map(silo_gpu_util_file):
    with open(silo_gpu_util_file, 'r') as f:
        gpu_util = yaml.load(f, Loader=yaml.FullLoader)
        gpu_util_map = {}
        i = 0
        logging.info("$$$$$$$$$$$$$$$$$$$")
        logging.info(gpu_util.items())

        for node, gpus_util_map_node in gpu_util.items():
            for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_node):
                logging.info(gpu_j)
                logging.info(num_process_on_gpu)
                for _ in range(num_process_on_gpu):
                    gpu_util_map[i] = (node, gpu_j)
                    i += 1
    print("GPU MAPPING: ", gpu_util_map)
    return gpu_util_map
