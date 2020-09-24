import logging

import torch
import torch.nn as nn


class MTLModel(nn.Module):

    def __init__(self, shared_nn, task_specific_nn=None):
        super(MTLModel, self).__init__()
        self.shared_NN = shared_nn
        self.task_specific_layer = shared_nn.fc

    def forward(self, x):
        x = self.shared_NN(x)
        return x
