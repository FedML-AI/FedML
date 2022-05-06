import MNN
import fedml
from fedml.cross_device import ServerMNN
from fedml.model import create_mnn_resnet20_model

from my_dataset import Cifar10Dataset

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    train_dataset = Cifar10Dataset(True)
    test_dataset = Cifar10Dataset(False)
    train_dataloader = MNN.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = MNN.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    class_num = 10

    # load model
    create_mnn_resnet20_model(args.global_model_file_path)

    # start training
    server = ServerMNN(
        args, device, test_dataloader, None
    )  # for MNN, the model is loaded using MNN file
    server.run()
