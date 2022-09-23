import fedml
from fedml import FedMLRunner

from fedml.model.cv.resnet56.resnet_pretrained import resnet56_pretrained


def create_model():
    # please download the pre-trained weight file from
    # https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt
    pre_trained_model_path = "./resnet56_on_cifar10.pt"
    model = resnet56_pretrained(10, pretrained=True, path=pre_trained_model_path)
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
