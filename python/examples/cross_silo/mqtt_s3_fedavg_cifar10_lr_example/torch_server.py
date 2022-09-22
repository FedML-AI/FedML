import fedml
from fedml import FedMLRunner
from fedml.data.data_loader import load
from trainer.classification_aggregator import ClassificationAggregator
from trainer.classification_trainer import ClassificationTrainer

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, class_num = load(args)

    # create model and trainer
    model = fedml.model.create(args, output_dim=class_num)
    trainer = ClassificationTrainer(model=model, args=args)
    aggregator = ClassificationAggregator(model=model, args=args)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
