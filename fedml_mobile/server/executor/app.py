import datetime
import logging
import os
import sys

import argparse
import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_mobile.server.executor.fedavg.FedAVGAggregator import FedAVGAggregator
from fedml_mobile.server.executor.fedavg.FedAvgServerManager import FedAVGServerManager
from fedml_mobile.server.executor.log import __log

from fedml_api.model.deep_neural_networks.mobilenet import mobilenet
from fedml_api.model.deep_neural_networks.resnet import resnet56

from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_distributed_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_distributed_cinic10
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_distributed_cifar10

from fedml_core.distributed.communication import Observer

from flask import Flask, request, jsonify


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
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

    parser.add_argument('--client_number', type=int, default=4, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=2, metavar='EP',
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


# HTTP server
app = Flask(__name__)

# parse python script input parameters
parser = argparse.ArgumentParser()
args = add_args(parser)

device_id_to_client_id_dict = dict()


@app.route('/api/register', methods=['POST'])
def register_device():
    global device_id_to_client_id_dict
    __log.info("register_device()")
    device_id = request.args['device_id']
    registered_client_num = len(device_id_to_client_id_dict)
    if device_id in device_id_to_client_id_dict:
        client_id = device_id_to_client_id_dict[device_id]
    else:
        client_id = registered_client_num + 1
        device_id_to_client_id_dict[device_id] = client_id

    training_task_args = {"dataset": args.dataset,
                          "data_dir": args.data_dir,
                          "partition_method": args.partition_method,
                          "partition_alpha": args.partition_alpha,
                          "model": args.model,
                          "client_number": args.client_number,
                          "comm_round": args.comm_round,
                          "epochs": args.epochs,
                          "lr": args.lr,
                          "wd": args.wd,
                          "batch_size": args.batch_size,
                          "frequency_of_the_test": args.frequency_of_the_test}

    return jsonify({"errno": 0,
                    "executorId": "executorId",
                    "executorTopic": "executorTopic",
                    "client_id": client_id,
                    "training_task_args": training_task_args})


if __name__ == '__main__':
    # MQTT client connection
    class Obs(Observer):
        def receive_message(self, msg_type, msg_params) -> None:
            global __log
            __log.info("receive_message(%s,%s)" % (msg_type, msg_params))


    logging.info(args)

    wandb.init(
        # project="federated_nas",
        project="fedml",
        name="FedAVG(d)" + str(args.partition_method) + "r" + str(args.comm_round) + "-e" + str(
            args.epochs) + "-lr" + str(
            args.lr),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(0)

    # GPU 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    train_data_local, test_data_local, class_num = data_loader(0, args.dataset, args.data_dir,
                                                               args.partition_method, args.partition_alpha,
                                                               args.client_number, args.batch_size)

    # create the model
    model = None
    if args.model == "resnet56":
        model = resnet56(class_num)
    elif args.model == "mobilenet":
        model = mobilenet(class_num=class_num)

    aggregator = FedAVGAggregator(train_data_global, test_data_global, train_data_num, args.client_number, device, model,
                                  args)
    server_manager = FedAVGServerManager(args, aggregator)
    server_manager.run()

    app.run(host='127.0.0.1', port=5000, debug=False)
