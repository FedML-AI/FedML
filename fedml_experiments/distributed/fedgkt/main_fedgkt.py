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
from mpi4py import MPI

# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from fedml_api.distributed.fedgkt.FedGKTAPI import FedML_init, FedML_FedGKT_distributed

from fedml_api.model.cv.resnet56_gkt.resnet_client import resnet8_56
from fedml_api.model.cv.resnet56_gkt.resnet_pretrained import resnet56_pretrained
from fedml_api.model.cv.resnet56_gkt.resnet_server import resnet56_server

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model_client', type=str, default='resnet5', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--model_server', type=str, default='resnet32', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')

    parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)

    parser.add_argument('--epochs_client', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--local_points', type=int, default=5000, metavar='LP',
                        help='the approximate fixed number of data points we will have on each local worker')

    parser.add_argument('--client_number', type=int, default=4, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--comm_round', type=int, default=300,
                        help='how many round of communications we shoud use')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--loss_scale', type=float, default=1024,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--no_bn_wd', action='store_true', help='Remove batch norm from weight decay')

    # knowledge distillation
    parser.add_argument('--temperature', default=3.0, type=float, help='Input the temperature: default(3.0)')
    parser.add_argument('--epochs_server', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained on the server side')
    parser.add_argument('--alpha', default=1.0, type=float, help='Input the relative weight: default(1.0)')
    parser.add_argument('--optimizer', default="SGD", type=str, help='optimizer: SGD, Adam, etc.')
    parser.add_argument('--whether_training_on_client', default=1, type=int)
    parser.add_argument('--whether_distill_on_the_server', default=0, type=int)
    parser.add_argument('--client_model', default="resnet4", type=str)
    parser.add_argument('--weight_init_model', default="resnet32", type=str)
    parser.add_argument('--running_name', default="default", type=str)
    parser.add_argument('--sweep', default=0, type=int)
    parser.add_argument('--multi_gpu_server', action='store_true')
    parser.add_argument('--test', action='store_true',
                        help='test mode, only run 1-2 epochs to test the bug of the program')

    parser.add_argument('--gpu_num_per_server', type=int, default=8,
                        help='gpu_num_per_server')

    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
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
                            args.partition_alpha, args.client_number, args.batch_size)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def create_client_model(args, n_classes):
    """
        Note that we only initialize the client feature extractor to mitigate the difficulty of alternating optimization
    """
    if args.dataset == "cifar10" or args.dataset == "CIFAR10":
        resumePath = "./../../../fedml_api/model/cv/pretrained/CIFAR10/resnet56/best.pth"
    elif args.dataset == "cifar100" or args.dataset == "CIFAR100":
        resumePath = "./../../../fedml_api/model/cv/pretrained/CIFAR100/resnet56/best.pth"
    elif args.dataset == "cinic10" or args.dataset == "CINIC10":
        resumePath = "./../../../fedml_api/model/cv/pretrained/CINIC10/resnet56/best.pth"
    else:
        resumePath = "./../../../fedml_api/model/cv/pretrained/CIFAR10/resnet56/best.pth"
    pretrained_model = resnet56_pretrained(n_classes, pretrained=True, path=resumePath)
    logging.info("########pretrained model#################")
    logging.info(pretrained_model)

    # copy pretrained parameters to client models
    params_featrue_extractor = dict()
    for name, param in pretrained_model.named_parameters():
        if name.startswith("conv1") or name.startswith("bn1") or name.startswith("layer1"):
            logging.info(name)
            params_featrue_extractor[name] = param

    client_model = resnet8_56(n_classes)

    logging.info("pretrained:")
    for name, param in client_model.named_parameters():
        if name.startswith("conv1"):
            param.data = params_featrue_extractor[name]
            if args.whether_training_on_client == 0:
                param.requires_grad = False
        elif name.startswith("bn1"):
            param.data = params_featrue_extractor[name]
            if args.whether_training_on_client == 0:
                param.requires_grad = False
        elif name.startswith("layer1"):
            param.data = params_featrue_extractor[name]
            if args.whether_training_on_client == 0:
                param.requires_grad = False
    logging.info(client_model)
    return client_model


def create_server_model(n_classes):
    server_model = resnet56_server(n_classes)
    return server_model


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

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # customize the process name
    str_process_name = "FedAvg (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            project="knowledge-distillation",
            name="FedGKT-" + str(args.running_name),
            config=args,
            tags="group_knowledge_transfer"
        )

    # Set the random seed if provided (affects client sampling, and batching)
    # if a pseudorandom number generator is reinitialized with the same seed
    # it will produce the same sequence of numbers.
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(size))

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

    # load data.
    # Note: if you use # of client epoch larger than 1,
    # please set the shuffle=False for the dataloader (CIFAR10/CIFAR100/CINIC10),
    # which keeps the batch sequence order across epoches.
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model
    if process_id == 0:
        model = create_server_model(class_num)
    else:
        model = create_client_model(args, class_num)

    # start distributed training
    FedML_FedGKT_distributed(process_id, worker_number, device, comm, model, train_data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, args)
