import argparse
import logging
import os

import numpy as np
import torch
import wandb
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.standalone.fedavg.data_loader import partition_data, get_dataloader
from fedml_api.standalone.fedavg.fedavg_trainer import FedAvgTrainer

args_datadir = "/home/chaoyanghe/sourcecode/dataset/cv/CIFAR10"
args_logdir = "log/cifar10"
args_alpha = 0.5
args_net_config = [3072, 100, 10]

switch_wandb = True


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--partition', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')

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
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(10)

    # data
    # input: args.dataset, args.data_dir
    logger.info("Partitioning data")
    # input:
    # output:
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset, args_datadir,
                                                                                             args_logdir,
                                                                                             args.partition,
                                                                                             args.client_number,
                                                                                             args_alpha,
                                                                                             args=args)
    train_dl_global, test_dl_global = get_dataloader(args.dataset, args_datadir, args.batch_size, 32)

    n_classes = len(np.unique(y_train))
    print("n_classes = " + str(n_classes))
    print("traindata_cls_counts = " + str(traindata_cls_counts))
    print("train_dl_global number = " + str(len(train_dl_global)))
    print("test_dl_global number = " + str(len(test_dl_global)))

    trainer = FedAvgTrainer(net_dataidx_map, train_dl_global, test_dl_global, device, args, n_classes, logger,
                            switch_wandb)
    trainer.train()
    trainer.global_test()
