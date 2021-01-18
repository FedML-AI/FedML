import argparse
import logging
import os
import random
import socket
import sys
import yaml

import traceback
from mpi4py import MPI

import numpy as np
import psutil
import setproctitle
import torch
import wandb
# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
from fedml_api.data_preprocessing.stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from fedml_api.data_preprocessing.stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from fedml_api.data_preprocessing.ImageNet.data_loader import load_partition_data_ImageNet
from fedml_api.data_preprocessing.Landmarks.data_loader import load_partition_data_landmarks

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10

from fedml_api.model.cv.cnn import CNN_DropOut
from fedml_api.model.cv.resnet_gn import resnet18
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.cv.mobilenet_v3 import MobileNetV3
from fedml_api.model.cv.efficientnet import EfficientNet


from fedml_api.utils.context import (
    raise_MPI_error
)
from fedml_api.utils.logger import (
    logging_config
)

from fedml_api.distributed.fedavg.FedAvgAPI import FedML_init, FedML_FedAvg_distributed


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """

    # wandb settings
    parser.add_argument('--entity', type=str, default=None,
                        help='Indicate the wandb entity that you want to sync')
    parser.add_argument('--project', type=str, default='fedml',
                        help='Indicate the wandb project that you want to sync')

    # algorithm settings
    parser.add_argument('--algorithm', type=str, default='fedavg',
                        help='Indicate the algorithm that you want to run')

    # Training settings
    parser.add_argument('--model', type=str, default='mobilenet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    # parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--gpu_util_file', type=str, default=None,
                        help='the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_util_key', type=str, default=None,
                        help='the key in gpu utilization file')

    # logging settings
    parser.add_argument('--level', type=str, default='INFO',
                        help='level of logging')

    # Optimizer parameters
    # parser.add_argument('--opt', type=str, default='sgd',
    #                     help='SGD with momentum; adam')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--wd', help='weight decay parameter;',
                        type=float, default=0.0001)
    parser.add_argument('--momentum', help='weight decay parameter;',
                        type=float, default=0.0)
    parser.add_argument('--nesterov', help='weight decay parameter;',
                        action='store_true', default=False)
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='StepLR', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--step-size', type=int, default=1, metavar='RATE',
                        help='LR decay gap (default: 1)')
    parser.add_argument('--lr-milestones', type=list, default=[30, 60], metavar='RATE',
                        help='LR decay rate milestones (default: 0.1)')
    parser.add_argument('--lr-T-max', type=int, default=10, metavar='RATE',
                        help='CosineAnnealingLR parameters (default: 0.1)')
    parser.add_argument('--lr-eta-min', type=float, default=0, metavar='RATE',
                        help='CosineAnnealingLR parameters (default: 0.1)')



    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size)
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "ILSVRC2012":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_ImageNet(dataset=dataset_name, data_dir=args.data_dir,
            partition_method=None, partition_alpha=None, 
            client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_dir, 'mini_gld_train_split.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'mini_gld_test.csv')
        args.data_dir = os.path.join(args.data_dir, 'images')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
            fed_train_map_file=fed_train_map_file, fed_test_map_file=fed_test_map_file,
            partition_method=None, partition_alpha=None, 
            client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_dir, 'federated_train.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'test.csv')
        args.data_dir = os.path.join(args.data_dir, 'images')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
            fed_train_map_file=fed_train_map_file, fed_test_map_file=fed_test_map_file,
            partition_method=None, partition_alpha=None, 
            client_number=args.client_num_in_total, batch_size=args.batch_size)

    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size)
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
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
    # TODO
    elif model_name == 'mobilenet_v3':
        '''model_mode \in {LARGE: 5.15M, SMALL: 2.94M}'''
        model = MobileNetV3(model_mode='LARGE', num_classes=output_dim)
    elif model_name == 'efficientnet':
        # model = EfficientNet()
        efficientnet_dict = {
            # Coefficients:   width,depth,res,dropout
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5),
            'efficientnet-b8': (2.2, 3.6, 672, 0.5),
            'efficientnet-l2': (4.3, 5.3, 800, 0.5),
        }
        # default is 'efficientnet-b0'
        model = EfficientNet.from_name(
            model_name='efficientnet-b0', num_classes=output_dim)
        # model = EfficientNet.from_pretrained(model_name='efficientnet-b0')
    else:
        raise NotImplementedError

    return model


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


def init_training_device_from_gpu_util_file(process_id, worker_number, gpu_util_file, gpu_util_key):

    if gpu_util_file == None:
        device = torch.device("cpu")
        logging.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(" ##################  Not Indicate gpu_util_file, using cpu  #################")
        logging.info(device)
        #return gpu_util_map[process_id][1]
        return device
    else:
        with open(gpu_util_file, 'r') as f:
            gpu_util_yaml = yaml.load(f, Loader=yaml.FullLoader)
            # gpu_util_num_process = 'gpu_util_' + str(worker_number)
            # gpu_util = gpu_util_yaml[gpu_util_num_process]
            gpu_util = gpu_util_yaml[gpu_util_key]
            gpu_util_map = {}
            i = 0
            for host, gpus_util_map_host in gpu_util.items():
                for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
                    for _ in range(num_process_on_gpu):
                        gpu_util_map[i] = (host, gpu_j)
                        i += 1
            logging.info("Process %d running on host: %s,gethostname: %s, gpu: %d ..." % (
                process_id, gpu_util_map[process_id][0], socket.gethostname(), gpu_util_map[process_id][1]))
            assert i == worker_number

        device = torch.device("cuda:" + str(gpu_util_map[process_id][1]) if torch.cuda.is_available() else "cpu")
        logging.info(device)
        #return gpu_util_map[process_id][1]
        return device

if __name__ == "__main__":
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    with raise_MPI_error():
        # parse python script input parameters
        parser = argparse.ArgumentParser()
        args = add_args(parser)
        args.rank = process_id
        logging.info(args)

        # customize the process name
        str_process_name = args.algorithm + " (distributed):" + str(process_id)
        setproctitle.setproctitle(str_process_name)

        logging_config(args, process_id)

        hostname = socket.gethostname()
        logging.info("#############process ID = " + str(process_id) +
                    ", host name = " + hostname + "########" +
                    ", process ID = " + str(os.getpid()) +
                    ", process Name = " + str(psutil.Process(os.getpid())))


        # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
        if process_id == 0:
            wandb.init(
                entity=args.entity,
                project=args.project,
                name=args.algorithm + " (d)" + str(args.partition_method) + "-" +str(args.dataset)+
                    "-r" + str(args.comm_round) +
                    "-e" + str(args.epochs) + "-" + str(args.model) + "-" +
                    str(args.client_optimizer) + "-bs" + str(args.batch_size) +
                    "-lr" + str(args.lr) + "-wd" + str(args.wd),
                config=args
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
        logging.info("process_id = %d, size = %d" % (process_id, worker_number))
        device = init_training_device_from_gpu_util_file(process_id, worker_number, args.gpu_util_file, args.gpu_util_key)

        # load data
        dataset = load_data(args, args.dataset)
        [train_data_num, test_data_num, train_data_global, test_data_global,
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

        # create model.
        # Note if the model is DNN (e.g., ResNet), the training will be very slow.
        # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
        model = create_model(args, model_name=args.model, output_dim=dataset[7])


        if args.algorithm == 'FedAvg':
            FedML_FedAvg_distributed(process_id, worker_number, device, comm,
                                    model, train_data_num, train_data_global, test_data_global,
                                    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args)
        else:
            raise NotImplementedError







