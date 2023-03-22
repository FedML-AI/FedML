import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

import fedml
from fedml import FedMLRunner
from fedml.arguments import Arguments
from fedml.model.cv.resnet import resnet20
from fedml.model.cv.resnet_cifar import resnet18_cifar
from fedml.model.cv.resnet_gn import resnet18
from fedml.model.cv.resnet_torch import resnet18 as resnet18_torch
from fedml.model.cv.cnn import Cifar10FLNet




def str2bool(v):
    if isinstance(v, bool):
        return v
    # if v.lower() in ('yes', 'true', 't', 'y', '1'):
    if isinstance(v, str) and v.lower() in ('true', 'True'):
        return True
    elif isinstance(v, str) and v.lower() in ('false', 'False'):
        return False
    else:
        return v
        # raise argparse.ArgumentTypeError('Boolean value expected.')



def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file", "--cf", help="yaml configuration file", type=str, default="",
    )

    # default arguments
    parser.add_argument("--run_id", type=str, default="0")
    parser.add_argument("--gpu_id", type=int, default=0)

    # default arguments
    parser.add_argument("--rank", type=int, default=0)

    # default arguments
    parser.add_argument("--local_rank", type=int, default=0)

    # For hierarchical scenario
    parser.add_argument("--node_rank", type=int, default=0)

    # default arguments
    parser.add_argument("--role", type=str, default="client")

    # cluster arguments
    parser.add_argument("--worker_num", type=int, default=8)
    parser.add_argument("--gpu_util_parse", type=str, default="localhost:2,1,1,1,1,1,1,1")

    # Model and dataset
    parser.add_argument("--model", type=str, default="resnet18_cifar")
    parser.add_argument("--group_norm_channels", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="fed_cifar100")
    parser.add_argument(
        "--data_cache_dir", type=str, default="/home/chaoyanghe/zhtang_FedML/python/fedml/data/fed_cifar100/datasets"
    )

    # Training arguments
    parser.add_argument("--federated_optimizer", type=str, default="FedAvg_seq")
    parser.add_argument("--client_num_in_total", type=int, default=500)
    parser.add_argument("--client_num_per_round", type=int, default=10)
    parser.add_argument("--comm_round", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--client_optimizer", type=str, default="sgd")
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--server_optimizer", type=str, default="sgd")
    parser.add_argument("--server_lr", type=float, default=1.0)
    parser.add_argument("--server_momentum", type=float, default=0.9)
    # args, unknown = parser.parse_known_args()

    parser.add_argument("--mimelite", type=str, default="False")

    #
    parser.add_argument("--feddyn_alpha", type=float, default=0.01)

    parser.add_argument("--initialize_all_clients", type=str, default="False")
    parser.add_argument("--cache_client_status", type=str, default="False")
    parser.add_argument("--override_cmd_args", action="store_true")

    args = parser.parse_args()

    for key in args.__dict__.keys():
        args.__dict__[key] = str2bool(args.__dict__[key])

    return args


def load_arguments(training_type=None, comm_backend=None):
    cmd_args = add_args()
    logging.info(f"cmd_args: {cmd_args}")

    # Load all arguments from YAML config file
    args = Arguments(cmd_args, training_type, comm_backend, override_cmd_args=cmd_args.override_cmd_args)

    # if not hasattr(args, "worker_num"):
    #     args.worker_num = args.client_num_per_round

    # os.path.expanduser() method in Python is used
    # to expand an initial path component ~( tilde symbol)
    # or ~user in the given path to user’s home directory.
    if hasattr(args, "data_cache_dir"):
        args.data_cache_dir = os.path.expanduser(args.data_cache_dir)
    if hasattr(args, "data_file_path"):
        args.data_file_path = os.path.expanduser(args.data_file_path)
    if hasattr(args, "partition_file_path"):
        args.partition_file_path = os.path.expanduser(args.partition_file_path)
    if hasattr(args, "part_file"):
        args.part_file = os.path.expanduser(args.part_file)

    args.rank = int(args.rank)
    return args


if __name__ == "__main__":
    # init FedML framework
    args = load_arguments(fedml._global_training_type, fedml._global_comm_backend)
    logging.info(f"args: {args}")
    args = fedml.init(args)
    logging.info(f"args: {args}")
    # args = fedml.init()

    # init device
    device = fedml.device.get_device(args)
    logging.info(
        f"======================================================== \
        device: {device} =============="
    )

    # load data
    dataset, output_dim = fedml.data.load(args)

    if args.dataset == "femnist":
        in_channels = 1
    else:
        in_channels = 3

    # load model (the size of MNIST image is 28 x 28)
    if args.model == "resnet18":
        logging.info("ResNet18_GN")
        model = resnet18(group_norm=args.group_norm_channels, num_classes=output_dim)
    elif args.model == "resnet20":
        model = resnet20(class_num=output_dim)
    elif args.model == "resnet18_torch":
        model = resnet18_torch(num_classes=output_dim, in_channels=in_channels)
    elif args.model == "resnet18_cifar":
        logging.info("ResNet18_GN")
        model = resnet18_cifar(group_norm=args.group_norm_channels, num_classes=output_dim)
    elif args.model == "cifar10cnn":
        logging.info("cifar10cnn")
        model = Cifar10FLNet()
    else:
        model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
