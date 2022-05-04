import logging

from fedml.model.cv.cnn import CNN_DropOut
from fedml.model.cv.efficientnet import EfficientNet
from fedml.model.cv.mobilenet import mobilenet
from fedml.model.cv.mobilenet_v3 import MobileNetV3
from fedml.model.cv.resnet import resnet56
from fedml.model.cv.resnet_gn import resnet18
from fedml.model.linear.lr import LogisticRegression
from fedml.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow


def create(args, output_dim):
    model_name = args.model
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "cnn" and args.dataset == "mnist":
        logging.info("CNN + MNIST")
        model = CNN_DropOut(False)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == "mobilenet_v3":
        """model_mode \in {LARGE: 5.15M, SMALL: 2.94M}"""
        model = MobileNetV3(model_mode="LARGE")
    elif model_name == "efficientnet":
        model = EfficientNet()
    else:
        model = LogisticRegression(28 * 28, output_dim)
    return model
