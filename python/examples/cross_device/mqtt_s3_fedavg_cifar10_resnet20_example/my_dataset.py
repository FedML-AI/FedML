import MNN
from torchvision.datasets import CIFAR10

F = MNN.expr


class Cifar10Dataset(MNN.data.Dataset):
    def __init__(self, training_dataset=True):
        super(Cifar10Dataset, self).__init__()
        self.is_training_dataset = training_dataset
        trainset = CIFAR10(root="./data", train=True, download=True)
        testset = CIFAR10(root="./data", train=False, download=True)
        if self.is_training_dataset:
            self.data = trainset.data.transpose(0, 3, 1, 2) / 255.0
            self.labels = trainset.targets
        else:
            self.data = testset.data.transpose(0, 3, 1, 2) / 255.0
            self.labels = testset.targets

    def __getitem__(self, index):
        dv = F.const(
            self.data[index].flatten().tolist(), [3, 32, 32], F.data_format.NCHW
        )
        dl = F.const([self.labels[index]], [], F.data_format.NCHW, F.dtype.uint8)
        # first for inputs, and may have many inputs, so it's a list
        # second for targets, also, there may be more than one targets
        return [dv], [dl]

    def __len__(self):
        # size of the dataset
        if self.is_training_dataset:
            return 50000
        else:
            return 10000
