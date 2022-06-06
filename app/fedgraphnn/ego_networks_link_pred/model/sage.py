import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGENodeCLF(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(SAGENodeCLF, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout
        self.nclass = nclass

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()

        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid, nhid))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.post(x)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
