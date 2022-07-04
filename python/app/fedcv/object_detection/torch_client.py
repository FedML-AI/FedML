import fedml
import torch
from fedml.cross_silo import Client
from trainer.detection_trainer import DetectionTrainer
from utils.init_yolo import init_yolo


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # init yolo
    model, dataset, args = init_yolo(args=args, device=device)

    # trainer
    trainer = DetectionTrainer(model=model, args=args)

    # start training
    client = Client(args, device, dataset, model, trainer)
    client.run()
