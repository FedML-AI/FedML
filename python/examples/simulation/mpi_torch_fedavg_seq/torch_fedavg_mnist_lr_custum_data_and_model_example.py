import logging

import fedml
from fedml.model.cv.resnet import resnet20
from fedml.model.cv.resnet_gn import resnet18
from fedml.simulation import SimulatorMPI

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
    elif args.model == "resnet20":
        model = resnet20(class_num=output_dim)
    else:
        model = fedml.model.create(args, output_dim)

    # start training
    simulator = SimulatorMPI(args, device, dataset, model)
    simulator.run()
