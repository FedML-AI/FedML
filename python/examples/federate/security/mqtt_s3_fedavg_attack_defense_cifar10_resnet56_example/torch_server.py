import logging

import fedml
from fedml import FedMLRunner
from fedml.model.cv.resnet import resnet56


def create_model():
    # please download the pre-trained weight file from
    # https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt
    pre_trained_model_path = "./config/resnet56_on_cifar10.pth"
    model = resnet56(10, pretrained=True, path=pre_trained_model_path)
    logging.info("load pretrained model successfully")
    return model


if __name__ == "__main__":
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = create_model()

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
