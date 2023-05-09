import logging
import fedml
from fedml import FedMLRunner
from fedml.model.cv.resnet import resnet56


def create_model():
    """
    loading pretrained model...
    please download the pre-trained weight file from
    https://github.com/FedML-AI/FedML/blob/fedml_v0.6_before_fundraising/fedml_api/model/cv/pretrained/CIFAR10/resnet56/best.pth
    and rename the file to ``resnet56_on_cifar10.pth''
    """
    pre_trained_model_path = "config/resnet56_on_cifar10.pth"
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
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
