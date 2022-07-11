import logging
import fedml
from fedml.cross_silo import Client
from data import load_data
from model import create_model
from trainer import ClassificationTrainer


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, class_num = load_data(args)

    # create model and trainer
    model = create_model(args, args.model, output_dim=class_num)
    trainer = ClassificationTrainer(model=model, args=args)

    # start training
    client = Client(args, device, dataset, model, model_trainer=trainer)
    client.run()
