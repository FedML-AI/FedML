import logging

import torch
import torch.nn as nn


class MTLModel(nn.Module):

    def __init__(self, shared_nn, task_specific_nn):
        super(MTLModel, self).__init__()
        self.shared_NN = shared_nn
        self.task_specific_layer = task_specific_nn

    def forward(self, x):
        x = self.shared_NN(x)
        x = self.task_specific_layer(x)
        return x
