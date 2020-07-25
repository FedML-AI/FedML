import argparse
import logging
import os
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data
from fedml_api.model.deep_neural_networks.mobilenet import mobilenet
from fedml_api.model.deep_neural_networks.resnet import resnet56
from fedml_api.standalone.fedavg.fedavg_trainer import FedAvgTrainer


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

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--local_points', type=int, default=5000, metavar='LP',
                        help='the approximate fixed number of data points we will have on each local worker')

    parser.add_argument('--client_number', type=int, default=4, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    args = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        project="federated_nas",
        name="FedAVG-r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    np.random.seed(0)
    torch.manual_seed(10)

    # load data
    train_data_num, test_data_num, train_data_global, test_data_global, \
    data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_partition_data(args.dataset, args.data_dir, args.partition_method,
                                    args.partition_alpha, args.client_number, args.batch_size)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]

    # create model
    # create the model
    model = None
    if args.model == "resnet56":
        model = resnet56(class_num)
    elif args.model == "mobilenet":
        model = mobilenet(class_num=class_num)

    trainer = FedAvgTrainer(dataset, model, device, args)
    trainer.train()
    trainer.global_test()
