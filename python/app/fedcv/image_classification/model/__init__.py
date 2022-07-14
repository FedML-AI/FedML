import logging
from .efficientnet.efficientnet import EfficientNet
from .densenet import DenseNet, densenet121, densenet161, densenet169, densenet201
from .mobilenet_v3 import MobileNetV3
from .cnn import CNN


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    if model_name.lower() == "cnn":
        model = CNN(input_size=3 if not args.input_size else args.input_size, num_classes=output_dim)
    elif model_name.lower() == "densenet":
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

    return model
