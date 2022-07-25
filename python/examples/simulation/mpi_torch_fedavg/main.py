import argparse
import logging
import os
import random
import socket
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

import fedml
import torch
from fedml.simulation import SimulatorMPI

from fedml.model.cv.resnet_gn import resnet18
from fedml.model.cv.resnet import resnet20, resnet32, resnet44, resnet56


from fedml.arguments import Arguments

from fedml import FedMLRunner


# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')


# def add_args():
#     parser = argparse.ArgumentParser(description="FedML")
#     parser.add_argument(
#         "--yaml_config_file",
#         "--cf",
#         help="yaml configuration file",
#         type=str,
#         default="",
#     )

#     # default arguments
#     parser.add_argument("--run_id", type=str, default="0")

#     # default arguments
#     parser.add_argument("--rank", type=int, default=0)

#     # default arguments
#     parser.add_argument("--local_rank", type=int, default=0)

#     # default arguments
#     parser.add_argument("--role", type=str, default="client")

#     parser.add_argument("--dataset", type=str, default="cifar10")
#     parser.add_argument("--data_cache_dir", type=str, default="~/datasets/cifar10")
#     parser.add_argument("--partition_method", type=str, default="hetero")
#     parser.add_argument("--partition_alpha", type=float, default=0.5)

#     parser.add_argument("--model", type=str, default="resnet20")
#     parser.add_argument("--federated_optimizer", type=str, default="FedAvg_seq")
#     parser.add_argument("--client_num_in_total", type=int, default=100)
#     parser.add_argument("--client_num_per_round", type=int, default=10)
#     parser.add_argument("--comm_round", type=int, default=4000)
#     parser.add_argument("--epochs", type=int, default=1)


#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--train_batch_size", type=int, default=200)
#     parser.add_argument("--test_batch_size", type=int, default=200)
#     parser.add_argument("--client_optimizer", type=str, default="sgd")
#     parser.add_argument("--learning_rate", type=float, default=0.1)

#     parser.add_argument("--frequency_of_the_test", type=str, default=5)
#     parser.add_argument("--role", type=str, default="client")

#     parser.add_argument("--worker_num", type=int, default=8)
#     parser.add_argument("--gpu_util_parse", type=str, default="localhost:2,1,1,1,1,1,1,1")

#     parser.add_argument("--enable_wandb", type=str2bool, default=True)
#     parser.add_argument("--wandb_key", type=str, default="ee0b5f53d949c84cee7decbe7a629e63fb2f8408")
#     parser.add_argument("--wandb_project", type=str, default="fedml")
#     parser.add_argument("--wandb_name", type=str, default="fedml_torch_fedavg")

#     args, unknown = parser.parse_known_args()
#     return args



# def load_arguments(training_type=None, comm_backend=None):
#     cmd_args = add_args()
#     # Load all arguments from YAML config file
#     # args = Arguments(cmd_args, training_type, comm_backend, override_cmd_args=False)
#     args = Arguments(cmd_args, training_type, comm_backend, override_cmd_args=True)
#     return args



if __name__ == "__main__":
    # init FedML framework
    # args = load_arguments(fedml._global_training_type, fedml._global_comm_backend)
    # args = fedml.init(args)
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)
    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model (the size of MNIST image is 28 x 28)
    if args.model == "resnet18":
        logging.info("ResNet18_GN")
        model = resnet18(group_norm=32)
    elif args.model == "resnet20":
        model = resnet20(class_num=output_dim)
    else:
        model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
    # simulator = SimulatorMPI(args, device, dataset, model)
    # simulator.run()



