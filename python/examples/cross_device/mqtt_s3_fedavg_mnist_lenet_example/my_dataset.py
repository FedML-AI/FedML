import MNN
from torchvision.datasets import MNIST

F = MNN.expr


class MnistDataset(MNN.data.Dataset):
    def __init__(self, datapath, training_dataset=True):
        super(MnistDataset, self).__init__()
        self.is_training_dataset = training_dataset
        trainset = MNIST(root=datapath, train=True, download=True)
        testset = MNIST(root=datapath, train=False, download=True)
        if self.is_training_dataset:
            self.data = trainset.data / 255.0
            self.labels = trainset.targets
        else:
            self.data = testset.data / 255.0
            self.labels = testset.targets

    def __getitem__(self, index):
        dv = F.const(
            self.data[index].flatten().tolist(), [1, 28, 28], F.data_format.NCHW
        )
        dl = F.const([self.labels[index]], [], F.data_format.NCHW, F.dtype.uint8)
        # first for inputs, and may have many inputs, so it's a list
        # second for targets, also, there may be more than one targets
        return [dv], [dl]

    def __len__(self):
        # size of the dataset
        if self.is_training_dataset:
            return 60000
        else:
            return 10000
