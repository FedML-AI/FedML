import argparse
import logging
import os
import random
import socket
import sys
import traceback

import numpy as np
import psutil
import setproctitle
import torch
import wandb
from mpi4py import MPI

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist

from fedml_api.model.cv.mnist_gan import MNIST_gan

from fedml_api.distributed.fedgan.FedGanAPI import FedML_init, FedML_FedGan_distributed
from fedml_api.distributed.fedgan.MyModelTrainer import MyModelTrainer


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='mnist_gan', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='mnist', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/mnist',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=3, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=3, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--backend', type=str, default="MPI",
                        help='Backend for Server and Client')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

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

    parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml",
                        help='the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str, default="mapping_default",
                        help='the key in gpu utilization file')

    parser.add_argument('--grpc_ipconfig_path', type=str, default="grpc_ipconfig.csv",
                        help='config table containing ipv4 address of grpc server')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5],
                                                             std=[0.5])]
                                       )
        mnist = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
        data_loader = DataLoader(dataset=mnist, batch_size=args.batch_size, shuffle=True, drop_last=True)
        # client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        # train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        # class_num = load_partition_data_mnist(args.batch_size)
        """
        For shallow NN or linear models,
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """

        test_data_num = 0
        train_data_local_dict = dict()
        test_data_local_dict = dict()
        train_data_local_num_dict = dict()
        train_data_global = list()
        test_data_global = list()
        class_num = 10

        for i in range(args.client_num_in_total):
            train_data_local_num_dict[i] = len(data_loader) * args.batch_size
            train_data_local_dict[i] = data_loader

        train_data_num = len(data_loader)

        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
        return dataset


def create_model():
    model = MNIST_gan()
    return model


if __name__ == "__main__":
    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    # if sys.platform == 'darwin':
    #     os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # customize the process name
    str_process_name = "FedGan (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO,
    # logging.basicConfig(level=logging.DEBUG,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()

    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Please check "GPU_MAPPING.md" to see how to define the topology
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, None,
                                                            args.gpu_mapping_key)

    # load data
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model()
    netd = model.get_netd()
    netg = model.get_netg()
    trainer = MyModelTrainer(netd, netg)

    # try:
        # start "federated averaging (FedAvg)"
    FedML_FedGan_distributed(process_id, worker_number, device, comm,
                             model, train_data_num, train_data_global, test_data_global,
                             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args, trainer)

    # except Exception as e:
    #     print(e)
    #     logging.info('traceback.format_exc():\n%s' % traceback.format_exc())
    #     MPI.COMM_WORLD.Abort()