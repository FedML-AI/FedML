
import logging

import fedml
from fedml.cross_device import ServerMNN
from my_dataset import MnistDataset

import MNN
F = MNN.expr

class MNNLenet5(MNN.nn.Module):
    """construct a lenet 5 model"""

    def __init__(self):
        super(MNNLenet5, self).__init__()
        self.conv1 = MNN.nn.conv(1, 20, [5, 5])
        self.conv2 = MNN.nn.conv(20, 50, [5, 5])
        self.fc1 = MNN.nn.linear(800, 500)
        self.fc2 = MNN.nn.linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool(x, [2, 2], [2, 2])
        x = F.relu(self.conv2(x))
        x = F.max_pool(x, [2, 2], [2, 2])
        # MNN use NC4HW4 format for convs, so we need to convert it to NCHW before entering other ops
        x = F.convert(x, F.NCHW)
        x = F.reshape(x, [0, -1])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, 1)
        return x


def create_mnn_lenet5_model(mnn_file_path):
    net = MNNLenet5()
    input_var = MNN.expr.placeholder([1, 1, 28, 28], MNN.expr.NCHW)
    predicts = net.forward(input_var)
    F.save([predicts], mnn_file_path)

import torch
import torch.nn as nn


class TorchLeNet(nn.Module):
    def __init__(self):
        super(TorchLeNet, self).__init__()
        self.fc2 = nn.Linear(500, 10)
        self.fc1 = nn.Linear(800, 500)
        self.dp = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(20, 50, (5, 5))
        self.conv1 = nn.Conv2d(1, 20, (5, 5))
        self.maxp = nn.MaxPool2d([2, 2], stride=(2, 2))
        self.rl = nn.ReLU()
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp(x)
        x = self.conv2(x)
        x = self.maxp(x)
        # add a reshape
        x = torch.reshape(x, (1, -1))
        x = self.fc1(x)
        x = self.rl(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = self.sm(x)
        return x

def create_model(args, output_dim):
    if args.model == "lenet" and hasattr(args, "deeplearning_backend") and args.deeplearning_backend == "mnn":
        create_mnn_lenet5_model(args.global_model_file_path)

        model = TorchLeNet()
    else:
        model = None
    # elif model_name == "resnet20" and hasattr(args, "deeplearning_backend") and args.deeplearning_backend == "mnn":
    #     from .mobile.mnn_resnet import create_mnn_resnet20_model

    #     create_mnn_resnet20_model(args.global_model_file_path)
    #     model = None  # for server MNN, the model is saved as computational graph and then send it to clients.
    return model
        
if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    train_dataset = MnistDataset(args.data_cache_dir, True)
    test_dataset = MnistDataset(args.data_cache_dir, False)
    train_dataloader = MNN.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = MNN.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    class_num = 10

    # load model
    model = create_model(args, output_dim=class_num)
    logging.info("FedMLDebug. model = {}".format(model))

    # start training
    server = ServerMNN(args, device, test_dataloader, model)
    server.run()


