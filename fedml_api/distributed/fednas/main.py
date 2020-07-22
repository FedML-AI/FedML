import argparse
import logging
import os
import socket

import numpy as np
import psutil
import setproctitle
import torch
import wandb
from client.client_manager import ClientMananger
from data_preprocessing.data_loader import partition_data, get_dataloader
from mpi4py import MPI
from server.server_manager import ServerMananger

# https://nyu-cds.github.io/python-mpi/05-collectives/
from model.FedNASAggregator import FedNASAggregator
from model.FedNASTrainer import FedNASTrainer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--stage', type=str, default='search',
                        help='stage: search; train')
    parser.add_argument('--model', type=str, default='resnet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--partition', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')

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


def init_server(args, comm, rank, size, round_num):
    # machine learning experiment tracking platform: https://www.wandb.com/
    wandb.init(
        project="federated_nas",
        name="FedNAS(d)" + str(args.partition) + "r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(
            args.learning_rate),
        config=args
    )

    # load data
    logging.info("load dataset")
    args_datadir = "./data/cifar10"
    args_logdir = "log/cifar10"
    args_alpha = 0.5
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset,
                                                                                             args_datadir,
                                                                                             args_logdir,
                                                                                             args.partition,
                                                                                             args.client_number,
                                                                                             args_alpha,
                                                                                             args=args)
    n_classes = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    logging.info("net_dataidx_map = " + str(net_dataidx_map))
    logging.info("#####################")

    all_train_data_num = sum([len(net_dataidx_map[r]) for r in range(args.client_number)])
    train_global, test_global = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size)
    logging.info("train_dl_global number = " + str(len(train_global)))
    logging.info("test_dl_global number = " + str(len(test_global)))

    # aggregator
    client_num = size - 1
    aggregator = FedNASAggregator(train_global, test_global, all_train_data_num, client_num, device, args)

    # start the distributed training
    server_manager = ServerMananger(args, comm, rank, size, round_num, aggregator)
    server_manager.run()


def init_client(args, comm, rank, size, round_num, seed):
    # to make sure each client has the same initial weight
    torch.manual_seed(seed)

    client_ID = rank - 1

    # 1. load data
    logging.info("load dataset")
    args_datadir = "./data/cifar10"
    args_logdir = "log/cifar10"
    args_alpha = 0.5
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset,
                                                                                             args_datadir,
                                                                                             args_logdir,
                                                                                             args.partition,
                                                                                             args.client_number,
                                                                                             args_alpha,
                                                                                             args=args)
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))

    all_train_data_num = sum([len(net_dataidx_map[r]) for r in range(args.client_number)])
    dataidxs = net_dataidx_map[client_ID]
    # logging.info("rank = %d, dataidxs = %s" % (rank, dataidxs))
    local_sample_number = len(dataidxs)
    logging.info("rank = %d, local_sample_number = %d" % (rank, local_sample_number))

    split = int(np.floor(0.5 * local_sample_number))  # split index
    train_idxs = dataidxs[0:split]
    test_idxs = dataidxs[split:local_sample_number]

    train_local, _ = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size, train_idxs)
    logging.info("rank = %d, batch_num_train_local = %d" % (rank, len(train_local)))

    test_local, _ = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size, test_idxs)
    logging.info("rank = %d, batch_num_test_local = %d" % (rank, len(test_local)))

    # 2. initialize the trainer
    trainer = FedNASTrainer(client_ID, train_local, test_local, local_sample_number, all_train_data_num, device, args)

    # 3. start the distributed training
    client_manager = ClientMananger(args, comm, rank, size, round_num, trainer)
    client_manager.run()


def init_training_device(process_ID, size):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    gpu_number = args.gpu

    for client_index in range(size - 1):
        gpu_index = client_index % gpu_number
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device


if __name__ == "__main__":

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

    # Set the random seed if provided (affects client sampling, and batching)
    # if a pseudorandom number generator is reinitialized with the same seed
    # it will produce the same sequence of numbers.
    seed = 0
    np.random.seed(seed)

    client_seed = np.random.randint(size)

    logging.info("rank = %d, size = %d" % (rank, size))
    device = init_training_device(rank, size)

    round_num = args.comm_round
    if rank == 0:
        init_server(args, comm, rank, size, round_num)
    else:
        init_client(args, comm, rank, size, round_num, client_seed)
