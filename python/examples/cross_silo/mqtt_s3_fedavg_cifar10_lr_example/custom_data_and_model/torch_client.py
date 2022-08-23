import fedml

from fedml import FedMLRunner
from trainer.classification_trainer import ClassificationTrainer
from trainer.classification_aggregator import ClassificationAggregator
import sys
sys.path.append('/Users/ranyide/Desktop/Python-Doc/FedML/python/fedml/data/')
import data_loader


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, class_num = data_loader.load(args)

    # create model and trainer
    model = fedml.model.create(args, output_dim=class_num)
    trainer = ClassificationTrainer(model=model, args=args)
    aggregator = ClassificationAggregator(model=model, args=args)


    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()