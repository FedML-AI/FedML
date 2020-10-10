import argparse
import logging
import os
import sys

import numpy as np
import setproctitle
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_distributed_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_distributed_cinic10
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_distributed_cifar10

from fedml_api.distributed.split_nn.SplitNNAPI import SplitNN_init, SplitNN_distributed
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet56', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_number', type=int, default=16, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--local_points', type=int, default=5000, metavar='LP',
                        help='the approximate fixed number of data points we will have on each local worker')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')
    args = parser.parse_args()
    return args


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


if __name__ == "__main__":
    comm, process_id, worker_number = SplitNN_init()

    parser = argparse.ArgumentParser()
    args = add_args(parser)

    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)

    str_process_name = "SplitNN (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(worker_number)

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
    train_data_local, test_data_local, class_num = data_loader(process_id, args.dataset, args.data_dir,
                                                               args.partition_method, args.partition_alpha,
                                                               args.client_number, args.batch_size)

    # create the model
    model = None
    split_layer = 1
    if args.model == "mobilenet":
        model = mobilenet(class_num=class_num)
    elif args.model == "resnet56":
        model = resnet56(class_num=class_num)

    fc_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Flatten(),
                             nn.Linear(fc_features, class_num))
    # Split The model
    client_model = nn.Sequential(*nn.ModuleList(model.children())[:split_layer])
    server_model = nn.Sequential(*nn.ModuleList(model.children())[split_layer:])

    SplitNN_distributed(process_id, worker_number, device, comm,
                        client_model, server_model, train_data_num,
                        train_data_global, test_data_global, local_data_num,
                        train_data_local, test_data_local, args)
