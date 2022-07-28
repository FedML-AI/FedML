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



