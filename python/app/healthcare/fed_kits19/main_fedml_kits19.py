import fedml
from fedml import FedMLRunner
from .data import load_data
from .model import create_model
from .trainer import create_trainer

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset = load_data(args)

    # create model and trainer
    model = create_model(args, args.model)
    trainer = create_trainer(model=model, args=args)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer)
    fedml_runner.run()
