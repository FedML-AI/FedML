import argparse
import logging
import os
import random
import socket
import sys

import numpy as np
import psutil
import setproctitle
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from fedml.centralized.centralized_trainer import CentralizedTrainer
from fedml.data.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from fedml.data.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml.data.fed_shakespeare.data_loader import (
    load_partition_data_federated_shakespeare,
)
from fedml.data.shakespeare.data_loader import load_partition_data_shakespeare
from fedml.data.stackoverflow_lr.data_loader import (
    load_partition_data_federated_stackoverflow_lr,
)
from fedml.data.stackoverflow_nwp.data_loader import (
    load_partition_data_federated_stackoverflow_nwp,
)
from fedml.data.MNIST.data_loader import load_partition_data_mnist
from fedml.data.ImageNet.data_loader import load_partition_data_ImageNet
from fedml.data.ImageNet.data_loader import distributed_centralized_ImageNet_loader
from fedml.data.Landmarks.data_loader import load_partition_data_landmarks

from fedml.data.cifar10.data_loader import load_partition_data_cifar10
from fedml.data.cifar100.data_loader import load_partition_data_cifar100
from fedml.data.cinic10.data_loader import load_partition_data_cinic10

from fedml.model.cv.cnn import CNN_DropOut
from fedml.model.cv.resnet_gn import resnet18
from fedml.model.cv.mobilenet import mobilenet
from fedml.model.cv.resnet import resnet56
from fedml.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow
from fedml.model.linear.lr import LogisticRegression
from fedml.model.cv.mobilenet_v3 import MobileNetV3
from fedml.model.cv.efficientnet import EfficientNet


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet",
        metavar="N",
        help="neural network used in training",
    )

    parser.add_argument(
        "--data_parallel", type=int, default=0, help="if distributed training"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_dir", type=str, default="./../../../data/cifar10", help="data directory"
    )

    parser.add_argument(
        "--partition_method",
        type=str,
        default="hetero",
        metavar="N",
        help="how to partition the dataset on local workers",
    )

    parser.add_argument(
        "--partition_alpha",
        type=float,
        default=0.5,
        metavar="PA",
        help="partition alpha (default: 0.5)",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=1000,
        metavar="NN",
        help="number of workers in a distributed cluster",
    )

    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=4,
        metavar="NN",
        help="number of workers",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--client_optimizer", type=str, default="adam", help="SGD with momentum; adam"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--wd", help="weight decay parameter;", type=float, default=0.001
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="EP",
        help="how many epochs will be trained locally",
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=10,
        help="how many round of communications we shoud use",
    )

    parser.add_argument(
        "--is_mobile",
        type=int,
        default=0,
        help="whether the program is running on the FedML-Mobile server side",
    )

    parser.add_argument(
        "--frequency_of_train_acc_report",
        type=int,
        default=10,
        help="the frequency of training accuracy report",
    )

    parser.add_argument(
        "--frequency_of_test_acc_report",
        type=int,
        default=1,
        help="the frequency of test accuracy report",
    )

    parser.add_argument("--gpu_server_num", type=int, default=1, help="gpu_server_num")

    parser.add_argument(
        "--gpu_num_per_server", type=int, default=4, help="gpu_num_per_server"
    )

    parser.add_argument("--ci", type=int, default=0, help="CI")

    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument("--gpu_util", type=str, default="0", help="gpu utils")

    parser.add_argument(
        "--local_rank", type=int, default=0, help="given by torch.distributed.launch"
    )

    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_mnist(args, args.batch_size)
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_emnist(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name in ["ILSVRC2012", "ILSVRC2012_hdf5"]:
        if args.data_parallel == 1:
            logging.info("load_data. dataset_name = %s" % dataset_name)
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = distributed_centralized_ImageNet_loader(
                dataset=dataset_name,
                data_dir=args.data_dir,
                world_size=args.world_size,
                rank=args.rank,
                batch_size=args.batch_size,
            )

        else:
            logging.info("load_data. dataset_name = %s" % dataset_name)
            (
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ) = load_partition_data_ImageNet(
                dataset=dataset_name,
                data_dir=args.data_dir,
                partition_method=None,
                partition_alpha=None,
                client_number=args.client_num_in_total,
                batch_size=args.batch_size,
            )

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(
            args.data_dir, "data_user_dict/gld23k_user_dict_train.csv"
        )
        fed_test_map_file = os.path.join(
            args.data_dir, "data_user_dict/gld23k_user_dict_test.csv"
        )
        args.data_dir = os.path.join(args.data_dir, "images")

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_landmarks(
            dataset=dataset_name,
            data_dir=args.data_dir,
            fed_train_map_file=fed_train_map_file,
            fed_test_map_file=fed_test_map_file,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(
            args.data_dir, "data_user_dict/gld160k_user_dict_train.csv"
        )
        fed_test_map_file = os.path.join(
            args.data_dir, "data_user_dict/gld160k_user_dict_test.csv"
        )
        args.data_dir = os.path.join(args.data_dir, "images")

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_landmarks(
            dataset=dataset_name,
            data_dir=args.data_dir,
            fed_train_map_file=fed_train_map_file,
            fed_test_map_file=fed_test_map_file,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = data_loader(
            args.dataset,
            args.data_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
        )
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset


def create_model(args, model_name, output_dim):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10004, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("CNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == "mobilenet_v3":
        """model_mode \in {LARGE: 5.15M, SMALL: 2.94M}"""
        model = MobileNetV3(model_mode="LARGE", num_classes=output_dim)
    elif model_name == "efficientnet":
        # model = EfficientNet()
        efficientnet_dict = {
            # Coefficients:   width,depth,res,dropout
            "efficientnet-b0": (1.0, 1.0, 224, 0.2),
            "efficientnet-b1": (1.0, 1.1, 240, 0.2),
            "efficientnet-b2": (1.1, 1.2, 260, 0.3),
            "efficientnet-b3": (1.2, 1.4, 300, 0.3),
            "efficientnet-b4": (1.4, 1.8, 380, 0.4),
            "efficientnet-b5": (1.6, 2.2, 456, 0.4),
            "efficientnet-b6": (1.8, 2.6, 528, 0.5),
            "efficientnet-b7": (2.0, 3.1, 600, 0.5),
            "efficientnet-b8": (2.2, 3.6, 672, 0.5),
            "efficientnet-l2": (4.3, 5.3, 800, 0.5),
        }
        # default is 'efficientnet-b0'
        model = EfficientNet.from_name(
            model_name="efficientnet-b0", num_classes=output_dim
        )

    return model


if __name__ == "__main__":

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.world_size = len(args.gpu_util.split(","))
    worker_number = 1
    process_id = 0

    if args.data_parallel == 1:
        # torch.distributed.init_process_group(
        #         backend="nccl", world_size=args.world_size, init_method="env://")
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.rank = torch.distributed.get_rank()
        gpu_util = args.gpu_util.split(",")
        gpu_util = [int(item.strip()) for item in gpu_util]
        # device = torch.device("cuda", local_rank)
        torch.cuda.set_device(gpu_util[args.rank])
        process_id = args.rank
    else:
        args.rank = 0

    logging.info(args)

    # customize the process name
    str_process_name = "Fedml (single):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(
        level=logging.INFO,
        # logging.basicConfig(level=logging.DEBUG,
        format=str(process_id)
        + " - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
    )
    hostname = socket.gethostname()
    logging.info(
        "#############process ID = "
        + str(process_id)
        + ", host name = "
        + hostname
        + "########"
        + ", process ID = "
        + str(os.getpid())
        + ", process Name = "
        + str(psutil.Process(os.getpid()))
    )

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="fedml",
            name="Fedml (central)"
            + str(args.partition_method)
            + "r"
            + str(args.comm_round)
            + "-e"
            + str(args.epochs)
            + "-lr"
            + str(args.lr),
            config=args,
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # GPU arrangement: Please customize this function according your own topology.
    # The GPU server list is configured at "mpi_host_file".
    # If we have 4 machines and each has two GPUs, and your FL network has 8 workers and a central worker.
    # The 4 machines will be assigned as follows:
    # machine 1: worker0, worker4, worker8;
    # machine 2: worker1, worker5;
    # machine 3: worker2, worker6;
    # machine 4: worker3, worker7;
    # Therefore, we can see that workers are assigned according to the order of machine list.
    logging.info("process_id = %d, size = %d" % (process_id, args.world_size))

    # load data
    dataset = load_data(args, args.dataset)
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])

    if args.data_parallel == 1:
        device = torch.device("cuda:" + str(gpu_util[args.rank]))
        model.to(device)
        model = DistributedDataParallel(
            model, device_ids=[gpu_util[args.rank]], output_device=gpu_util[args.rank]
        )
    else:
        device = torch.device(
            "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
        )
    # start "federated averaging (FedAvg)"
    single_trainer = CentralizedTrainer(dataset, model, device, args)
    single_trainer.train()
