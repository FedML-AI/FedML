import logging

import fedml
from fedml import FedMLRunner
from model import (
    densenet121,
    densenet161,
    densenet169,
    densenet201,
    MobileNetV3,
    EfficientNet,
)
from trainer.classification_trainer import ClassificationTrainer
from trainer.classification_aggregator import ClassificationAggregator
from data.data_loader import load_data


def create_model(args, model_name, output_dim):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    if model_name.lower() == "densenet":
        model = densenet121(num_classes=output_dim)
    elif model_name.lower() == "densenet121":
        model = densenet121(num_classes=output_dim)
    elif model_name.lower() == "densenet161":
        model = densenet161(num_classes=output_dim)
    elif model_name.lower() == "densenet169":
        model = densenet169(num_classes=output_dim)
    elif model_name.lower() == "densenet201":
        model = densenet201(num_classes=output_dim)
    elif model_name.lower() == "efficientnet":
        model = EfficientNet.from_name("efficientnet-l2", num_classes=output_dim)
    elif model_name.lower() == "mobilenetv3":
        model = MobileNetV3(num_classes=output_dim)
    else:
        raise Exception("such model does not exist !")

    logging.info("done")

    return model


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, class_num = load_data(args)

    # create model.
    model = create_model(args, args.model, output_dim=class_num)
    trainer = ClassificationTrainer(model=model, args=args)
    aggregator = ClassificationAggregator(model=model, args=args)

    logging.info("start training")

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
