import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class GIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout, eps):
        super(GIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(
            torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid)
        )
        self.graph_convs.append(GINConv(self.nn1, eps=eps))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(
                torch.nn.Linear(nhid, nhid),
                torch.nn.ReLU(),
                torch.nn.Linear(nhid, nhid),
            )
            self.graph_convs.append(GINConv(self.nnk, eps=eps))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
