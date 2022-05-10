import torch

from fedml.model.cv.densenet import DenseNet, densenet121, densenet161, densenet169, densenet201


class TestDenseNet:
    def test_densenet(self):
        DenseNet()
        densenet121()
        densenet161()
        densenet169()
        densenet201()
