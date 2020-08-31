import logging
import os
import sys
import time
from datetime import datetime

import argparse
import numpy as np
import requests
import torch

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.model.deep_neural_networks.mobilenet import mobilenet
from fedml_api.model.deep_neural_networks.resnet import resnet56
from fedml_mobile.server.executor.fedavg.FedAvgClientManager import FedAVGClientManager

from fedml_mobile.server.executor.fedavg.FedAVGTrainer import FedAVGTrainer
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_distributed_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_distributed_cinic10
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_distributed_cifar10


def add_args(parser):
    parser.add_argument('--client_uuid', type=str, default="0",
                        help='number of workers in a distributed cluster')
    args = parser.parse_args()
    return args


def register(uuid):
    str_device_UUID = uuid
    URL = "http://127.0.0.1:5000/api/register"

    # defining a params dict for the parameters to be sent to the API
    PARAMS = {'device_id': str_device_UUID}

    # sending get request and saving the response as response object
    r = requests.post(url=URL, params=PARAMS)
    result = r.json()
    client_ID = result['client_id']
    executorId = result['executorId']
    executorTopic = result['executorTopic']
    training_task_args = result['training_task_args']

    class Args:
        def __init__(self):
            self.dataset = training_task_args['dataset']
            self.data_dir = training_task_args['data_dir']
            self.partition_method = training_task_args['partition_method']
            self.partition_alpha = training_task_args['partition_alpha']
            self.model = training_task_args['model']
            self.client_number = training_task_args['client_number']
            self.comm_round = training_task_args['comm_round']
            self.epochs = training_task_args['epochs']
            self.lr = training_task_args['lr']
            self.wd = training_task_args['wd']
            self.batch_size = training_task_args['batch_size']
            self.frequency_of_the_test = training_task_args['frequency_of_the_test']

    args = Args()
    return client_ID, args


def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device


"""
python mobile_client_simulator.py --client_uuid '0'
python mobile_client_simulator.py --client_uuid '1'
python mobile_client_simulator.py --client_uuid '2'
python mobile_client_simulator.py --client_uuid '4'
"""
if __name__ == '__main__':
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    main_args = add_args(parser)
    uuid = main_args.client_uuid

    client_ID, args = register(uuid)
    logging.info("client_ID = " + str(client_ID))
    logging.info("dataset = " + str(args.dataset))
    logging.info("model = " + str(args.model))
    logging.info("client_number = " + str(args.client_number))

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(args.client_number)

    logging.info("client_ID = %d, size = %d" % (client_ID, args.client_number))
    device = init_training_device(client_ID-1, args.client_number - 1, 4)

    # load data
    if args.dataset == "cifar10":
        data_loader = load_partition_data_distributed_cifar10
    elif args.dataset == "cifar100":
        data_loader = load_partition_data_distributed_cifar100
    elif args.dataset == "cinic10":
        data_loader = load_partition_data_distributed_cinic10
    else:
        data_loader = load_partition_data_distributed_cifar10
    train_data_num, train_data_global, \
    test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = data_loader(client_ID, args.dataset, args.data_dir,
                                                               args.partition_method, args.partition_alpha,
                                                               args.client_number, args.batch_size)

    # create the model
    model = None
    if args.model == "resnet56":
        model = resnet56(class_num)
    elif args.model == "mobilenet":
        model = mobilenet(class_num=class_num)

    trainer = FedAVGTrainer(client_ID, train_data_local, local_data_num, train_data_num, device, model, args)

    client_manager = FedAVGClientManager(args, trainer)
    client_manager.update_sender_id(client_ID)
    client_manager.run()
    client_manager.start_training()

    time.sleep(100000)


