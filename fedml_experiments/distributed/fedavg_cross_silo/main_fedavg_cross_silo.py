#!/usr/bin/env python3
import json
import os
import platform
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from fedml_api.data_preprocessing.synthetic_1_1.data_loader import load_partition_data_federated_synthetic_1_1
from fedml_api.distributed.fedavg_cross_silo.FedAvgAPI import FedAvgAPI
from fedml_api.model.cv.efficientnet import EfficientNet
from fedml_api.model.cv.mobilenet_v3 import MobileNetV3
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow
from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet_gn import resnet18
from fedml_api.model.cv.cnn import CNN_DropOut
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.cifar10.data_loader_cross_silo import (
    load_partition_data_cifar10 as load_partition_data_cifar10_cross_silo,
)
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.Landmarks.data_loader import load_partition_data_landmarks
from fedml_api.data_preprocessing.ImageNet.data_loader import load_partition_data_ImageNet
from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from fedml_api.data_preprocessing.stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from fedml_api.data_preprocessing.stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
from fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist

from fedml_api.distributed.utils.gpu_mapping_mlops import mapping_processes_to_gpu_device_from_yaml_file

from fedml_api.model.mobile.moble_lenet import create_mobile_lenet_model

from fedml_api.distributed.fedavg_cross_silo.FedLogsSDK import FedLogsSDK

import argparse
import logging

import random
import socket


import numpy as np
import psutil
import setproctitle
import torch
import wandb
from mpi4py import MPI

env_dict = {
    "NCCL_IB_DISABLE": "1",
    "NCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "TP_SOCKET_IFNAME": "lo",
    "NCCL_DEBUG": "INFO",
    "NCCL_MIN_NRINGS": "1",
    "NCCL_TREE_THRESHOLD": "4294967296",
    "OMP_NUM_THREADS": "8",
    "NCCL_NSOCKS_PERTHREAD": "8",
    "NCCL_SOCKET_NTHREADS": "8",
    "NCCL_BUFFSIZE": "1048576",
    "WANDB_API_KEY": "ee0b5f53d949c84cee7decbe7a629e63fb2f8408",
}


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument("--model", type=str, default="mobilenet", metavar="N", help="neural network used in training")

    parser.add_argument(
        "--is_using_local_data", type=int, default=0, help="whether use the local private data"
    )

    parser.add_argument("--dataset", type=str, default="cifar10", metavar="N", help="dataset used for training")

    parser.add_argument("--data_dir", type=str, default="./../../../data/cifar10", help="data directory")


    parser.add_argument(
        "--partition_method",
        type=str,
        default="hetero",
        metavar="N",
        help="how to partition the dataset on local workers",
    )

    parser.add_argument(
        "--partition_alpha", type=float, default=0.5, metavar="PA", help="partition alpha (default: 0.5)"
    )

    parser.add_argument(
        "--data_silo_num_in_total",
        type=int,
        default=1000,
        metavar="NN",
        help="number of data silos in a distributed cluster",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=10,
        metavar="NN",
        help="number of clients in a distributed cluster",
    )

    parser.add_argument("--client_num_per_round", type=int, default=4, metavar="NN", help="number of workers")

    parser.add_argument(
        "--batch_size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )

    parser.add_argument("--client_optimizer", type=str, default="adam", help="SGD with momentum; adam")

    parser.add_argument("--backend", type=str, default="MPI", help="Backend for Server and Client")

    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")

    parser.add_argument("--wd", help="weight decay parameter;", type=float, default=0.0001)

    parser.add_argument("--epochs", type=int, default=5, metavar="EP", help="how many epochs will be trained locally")

    parser.add_argument("--comm_round", type=int, default=10, help="how many round of communications we shoud use")

    parser.add_argument(
        "--is_mobile", type=int, default=0, help="whether the program is running on the FedML-Mobile server side"
    )

    parser.add_argument("--frequency_of_the_test", type=int, default=1, help="the frequency of the algorithms")

    parser.add_argument("--gpu_server_num", type=int, default=1, help="gpu_server_num")

    parser.add_argument("--gpu_num_per_server", type=int, default=4, help="gpu_num_per_server")

    parser.add_argument(
        "--gpu_mapping_file",
        type=str,
        help="the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.",
    )

    parser.add_argument(
        "--gpu_mapping_key", type=str, default="mapping_default", help="the key in gpu utilization file"
    )

    parser.add_argument(
        "--grpc_ipconfig_path",
        type=str,
        default="grpc_ipconfig.csv",
        help="config table containing ipv4 address of grpc server",
    )

    parser.add_argument(
        "--trpc_master_config_path",
        type=str,
        default="trpc_master_config.csv",
        help="config indicating ip address and port of the master (rank 0) node",
    )

    parser.add_argument(
        "--enable_cuda_rpc",
        default=False,
        action="store_true",
        help="Enable cuda rpc (only for TRPC backend)",
    )

    parser.add_argument("--silo_node_rank", type=int, default=0, help="rank of the node in silo")

    parser.add_argument("--silo_rank", type=int, default=0, help="rank of the silo")

    # parser.add_argument(
    #     "--local_rank", type=int, default=1, help="local rank in the node, Passed by launcher.py"
    # )

    parser.add_argument("--nnode", type=int, default=1, help="number of nodes in silo")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="number of processes in each node")
    parser.add_argument("--pg_master_address", type=str, default=1, help="address of the DDP process group master")
    parser.add_argument("--pg_master_port", type=int, default=1, help="port of the DDP process group master")

    parser.add_argument(
        "--silo_gpu_mapping_file",
        type=str,
        help="the gpu utilization file for silo processes.",
    )

    # MQTT
    parser.add_argument(
        "--mqtt_config_path",
        type=str,
        help="Path of config for mqtt server.",
    )
    # --------------------------

    # S3
    parser.add_argument(
        "--s3_config_path",
        type=str,
        help="Path of config for S3 server.",
    )
    # --------------------------

    parser.add_argument(
        "--run_id",
        type=str,
        help="Run id for one federated training workflow.",
    )

    parser.add_argument(
        "--client_ids",
        type=str,
        help="Client id list in the same federated training run.",
    )

    parser.add_argument(
        "--client_objects",
        type=str,
        default="[]",
        help="Client object list in the same federated training run.",
    )

    parser.add_argument(
        "--synthetic_data_url",
        type=str,
        default="",
        help="Synthetic data URL from the run configuration which will be used as training data. " +
             " If you set this URL value, the system will not use local data as training data.",
    )

    parser.add_argument(
        "--log_file_dir",
        type=str,
        help="Log file directory.",
    )

    parser.add_argument(
        "--log_server_url",
        type=str,
        default="",
        help="Log server url.",
    )

    parser.add_argument("--ci", type=int, default=0, help="CI")
    args = parser.parse_args()
    return args


def load_synthetic_data(args, dataset_name):
    if dataset_name == "synthetic_1_1":
        (
            silo_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_synthetic_1_1()
        args.data_silo_num_in_total = silo_num
        # silo_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        # train_data_local_num_dict, train_data_local_dict, test_data_local_dict, output_dim = load_partition_data_federated_synthetic_1_1()
        print("test_data_loader finished")
    elif dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            silo_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_mnist(args.batch_size)
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.data_silo_num_in_total = silo_num

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            silo_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_emnist(args.dataset, args.data_dir)
        args.data_silo_num_in_total = silo_num

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            silo_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_shakespeare(args.batch_size)
        args.data_silo_num_in_total = silo_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            silo_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
        args.data_silo_num_in_total = silo_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            silo_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.data_silo_num_in_total = silo_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            silo_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
        args.data_silo_num_in_total = silo_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            silo_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        args.data_silo_num_in_total = silo_num
    elif dataset_name == "ILSVRC2012":
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
            client_number=args.data_silo_num_in_total,
            batch_size=args.batch_size,
        )

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.data_silo_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_dir, "mini_gld_train_split.csv")
        fed_test_map_file = os.path.join(args.data_dir, "mini_gld_test.csv")
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
            client_number=args.data_silo_num_in_total,
            batch_size=args.batch_size,
        )

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.data_silo_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_dir, "federated_train.csv")
        fed_test_map_file = os.path.join(args.data_dir, "test.csv")
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
            client_number=args.data_silo_num_in_total,
            batch_size=args.batch_size,
        )

    else:
        if dataset_name == "cifar10":
            # TODO: find a bette way for difference between clients and server
            if args.silo_rank == 0:
                logging.info("loading cifa10 for server")
                data_loader = load_partition_data_cifar10
            else:
                logging.info("loading cifa10 for client")
                data_loader = load_partition_data_cifar10_cross_silo
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
            args.data_silo_num_in_total,
            args.batch_size,
            args.silo_proc_num,
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
    args.model_file_path = None
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
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
    # TODO
    elif model_name == "mobilenet_v3":
        """model_mode \in {LARGE: 5.15M, SMALL: 2.94M}"""
        model = MobileNetV3(model_mode="LARGE")
    elif model_name == "efficientnet":
        model = EfficientNet()
    elif model_name == "lenet_mnist":
        model, args.model_file_path = create_mobile_lenet_model()
    elif model_name == "lr" and args.dataset == "synthetic_1_1":
        logging.info("lr + synthetic_1_1")
        model = LogisticRegression(60, output_dim)

    return model


def setup_default_environments():
    os_env = os.environ
    sys_name = platform.system()
    if sys_name == "Darwin":
        env_dict["NCCL_SOCKET_IFNAME"] = "lo0"
        env_dict["GLOO_SOCKET_IFNAME"] = "lo0"
        env_dict["TP_SOCKET_IFNAME"] = "lo0"
    else:
        env_dict["NCCL_SOCKET_IFNAME"] = "lo"
        env_dict["GLOO_SOCKET_IFNAME"] = "lo"
        env_dict["TP_SOCKET_IFNAME"] = "lo"

    for key in env_dict.keys():
        os_env[key] = env_dict[key]


if __name__ == "__main__":
    # setup environment variables
    setup_default_environments()

    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # parse python script input parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = add_args(parser)

    if args.enable_cuda_rpc and (not args.gpu_mapping_file):
        parser.error("Need to specify gpu_mapping for using cuda_rpc")

    if (args.backend == "MQTT" or args.backend == "MQTT_S3") and not args.mqtt_config_path:
        parser.error("Please add argument --mqtt_config_path")

    if args.backend == "MQTT_S3" and not args.mqtt_config_path:
        parser.error("Please add argument --s3_config_path")

    comm, process_id, worker_number = FedML_init()
    args.worker_silo_num = args.client_num_in_total + 1
    args.local_rank = process_id

    args.silo_proc_num = args.nnode * args.nproc_per_node
    args.silo_proc_rank = args.silo_node_rank * args.nproc_per_node + args.local_rank

    # # customize the process name
    str_process_name = "FedAvg (distributed):" + str(args.silo_rank)
    setproctitle.setproctitle(str_process_name)

    FedLogsSDK.get_instance(args).init_logs()
    logging.info(args)
    hostname = socket.gethostname()
    logging.info(
        "#############process ID = "
        + str(args.silo_rank)
        + ", host name = "
        + hostname
        + "########"
        + ", process ID = "
        + str(os.getpid())
        + ", process Name = "
        + str(psutil.Process(os.getpid()))
    )

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if args.silo_rank == 0 and args.silo_proc_rank == 0:
        print("wandb.init")
        wandb.init(
            project="mlops",
            entity="fedml-ai",
            name="FedAVG(d)"
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

    # Please check "GPU_MAPPING.md" to see how to define the topology
    logging.info(
        "silo_rank = %d, silo_proc_rank = %d, silo_proc_num = %d"
        % (args.silo_rank, args.silo_proc_rank, args.silo_proc_num)
    )

    device = mapping_processes_to_gpu_device_from_yaml_file(
        args.silo_rank, args.worker_silo_num, args.gpu_mapping_file, args.gpu_mapping_key, check_cross_silo=True
    )

    # load data
    if args.is_using_local_data == 0:
        dataset = load_synthetic_data(args, args.dataset)
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
    else:
        # TODO: @alex, please add the logic of loading private data
        args.data_silo_num_in_total = args.client_num_in_total
        raise Exception("need to add the logic of loading private data")

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])

    # print(len(train_data_local_num_dict.items()[0][0].items()))
    # print(len(train_data_local_num_dict.items()[0][0].))
    # start distributed training
    FedAvgAPI(
        args.silo_rank,
        args.worker_silo_num,
        device,
        None,
        model,
        train_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        args,
    )
