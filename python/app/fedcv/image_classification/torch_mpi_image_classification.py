import logging
import fedml
from data import load_data
from fedml.simulation import SimulatorMPI
from model import create_model
from trainer.classification_trainer import ClassificationTrainer


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
    try:
        simulator = SimulatorMPI(args, device, dataset, model, trainer)
        simulator.run()
    except Exception as e:
        raise e
