import logging

import argparse
from os import path

import fedml
from fedml import FedMLRunner
from fedml.model.cv.resnet_gn import resnet18
from fedml.model.cv.resnet import resnet20, resnet32, resnet44, resnet56

from fedml.arguments import Arguments



def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )

    # default arguments
    parser.add_argument("--run_id", type=str, default="0")

    # default arguments
    parser.add_argument("--rank", type=int, default=0)

    # default arguments
    parser.add_argument("--local_rank", type=int, default=0)

    # default arguments
    parser.add_argument("--role", type=str, default="client")

    args, unknown = parser.parse_known_args()
    return args



def load_arguments(training_type=None, comm_backend=None):
    cmd_args = add_args()
    # Load all arguments from YAML config file
    args = Arguments(cmd_args, training_type, comm_backend, override_cmd_args=False)
    return args



if __name__ == "__main__":
    # init FedML framework
    args = load_arguments(fedml._global_training_type, fedml._global_comm_backend)
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model (the size of MNIST image is 28 x 28)
    if args.model == "resnet18":
        logging.info("ResNet18_GN")
        model = resnet18()
    elif args.model == "resnet20":
        model = resnet20(class_num=output_dim)
    else:
        model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()




