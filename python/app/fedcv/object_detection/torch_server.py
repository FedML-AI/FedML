import logging
import fedml
import torch
from fedml.cross_silo import Server
from utils.init_yolo import init_yolo


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)
    logging.info("Device: {}".format(device))

    # init yolo
    model, dataset, trainer, args = init_yolo(args=args, device=device)
    logging.info("init model, dataset, trainer and args done")

    # start training
    server = Server(args, device, dataset, model, trainer)
    logging.info("init server done")
    server.run()
