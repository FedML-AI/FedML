import torch
import torch.nn.functional as F
from torch_geometric.nn import SGConv


class SGCNodeCLF(torch.nn.Module):
    def __init__(self, in_dim, num_classes, K):
        super(SGCNodeCLF, self).__init__()
        self.conv1 = SGConv(in_dim, num_classes, K)
        self.nclass = num_classes

    def forward(self, inp):
        x, edge_index = inp.x, inp.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
