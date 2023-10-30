import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNodeCLF(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, heads=2):
        super(GATNodeCLF, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, heads, heads=heads)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(heads * heads, out_channels, heads=1, concat=False, dropout=0.6)
        self.nclass = out_channels

    def forward(self, inp):

        x = F.elu(self.conv1(inp.x, inp.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, inp.edge_index)
        return F.log_softmax(x, dim=-1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
