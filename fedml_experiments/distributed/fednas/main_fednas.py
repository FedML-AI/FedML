import argparse
import logging
import os
import socket
import sys

import numpy as np
import psutil
import setproctitle
import torch
import wandb
# https://nyu-cds.github.io/python-mpi/05-collectives/
from mpi4py import MPI
# add the FedML root directory to the python path
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_distributed_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_distributed_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_distributed_cinic10
from fedml_api.distributed.fednas.FedNASAPI import FedML_init, FedML_FedNAS_distributed
from fedml_api.model.cv.darts import genotypes
from fedml_api.model.cv.darts.model import NetworkCIFAR
from fedml_api.model.cv.darts.model_search import Network

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')

    parser.add_argument('--stage', type=str, default='search',
                        help='stage: search; train')
    parser.add_argument('--model', type=str, default='resnet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--local_points', type=int, default=5000, metavar='LP',
                        help='the approximate fixed number of data points we will have on each local worker')

    parser.add_argument('--client_number', type=int, default=16, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--comm_round', type=int, default=50,
                        help='how many round of communications we shoud use')

    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='DARTS layers')

    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')

    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--lambda_train_regularizer', type=float, default=1, help='train regularizer parameter')
    parser.add_argument('--lambda_valid_regularizer', type=float, default=1, help='validation regularizer parameter')
    parser.add_argument('--report_freq', type=float, default=10, help='report frequency')

    parser.add_argument('--tau_max', type=float, default=10, help='initial tau')
    parser.add_argument('--tau_min', type=float, default=1, help='minimum tau')

    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--arch', type=str, default='FedNAS_V1', help='which architecture to use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1, help='the frequency of the test')
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
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    parser = argparse.ArgumentParser()
    args = add_args(parser)
    str_process_name = "Federated Learning:" + str(rank)

    setproctitle.setproctitle(str_process_name)

    logging.basicConfig(level=logging.INFO,
                        format=str(rank) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(rank) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="fednas_extension",
            name="FedNAS" + str(args.partition_method) + "r" + str(args.comm_round) + "-e" + str(
                args.epochs),
            config=args
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    np.random.seed(0)
    torch.manual_seed(worker_number)

    # GPU arrangement: Please customize this function according your own topology.
    # The GPU server list is configured at "mpi_host_file".
    # If we have 4 machines and each has two GPUs, and your FL network has 8 workers and a central worker.
    # The 4 machines will be assigned as follows:
    # machine 1: worker0, worker4, worker8;
    # machine 2: worker1, worker5;
    # machine 3: worker2, worker6;
    # machine 4: worker3, worker7;
    # Therefore, we can see that workers are assigned according to the order of machine list.
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)

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

    # create model
    model = None
    criterion = nn.CrossEntropyLoss().to(device)
    if args.stage == "search":
        model = Network(args.init_channels, class_num, args.layers, criterion, device)
    else:
        genotype = genotypes.FedNAS_V1
        logging.info(genotype)
        model = NetworkCIFAR(args.init_channels, class_num, args.layers, args.auxiliary, genotype)

    FedML_FedNAS_distributed(process_id, worker_number, device, comm,
                             model, train_data_num, train_data_global, test_data_global,
                             local_data_num, train_data_local, test_data_local, args)