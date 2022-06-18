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
from fedml.simulation import SimulatorNCCL
# from fedml import run_simulation

from fedml.model.cv.resnet_gn import resnet18


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model (the size of MNIST image is 28 x 28)
    if args.model == "resnet18":
        logging.info("ResNet18_GN")
        model = resnet18()
    else:
        model = fedml.model.create(args, output_dim)
    # start training
    simulator = SimulatorNCCL(args, device, dataset, model)
    # simulator = run_simulation(args, device, dataset, model)
    simulator.run()



