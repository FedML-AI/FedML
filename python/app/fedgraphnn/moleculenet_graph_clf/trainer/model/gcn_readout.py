import torch
import torch.nn as nn


class GCN(nn.Module):
    """
    Graph Convolutional Network based on https://arxiv.org/abs/1609.02907

    """

    def __init__(self, feat_dim, hidden_dim1, hidden_dim2, dropout, is_sparse=False):
        """Dense version of GAT."""
        super(GCN, self).__init__()
        # self.dropout = dropout
        self.W1 = nn.Parameter(torch.FloatTensor(feat_dim, hidden_dim1))
        self.W2 = nn.Parameter(torch.FloatTensor(hidden_dim1, hidden_dim2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.W1.data)
        nn.init.xavier_uniform_(self.W2.data)

        self.is_sparse = is_sparse

    def forward(self, x, adj):
        # Layer 1
        support = torch.mm(x, self.W1)
        embeddings = (
            torch.sparse.mm(adj, support) if self.is_sparse else torch.mm(adj, support)
        )

        embeddings = self.dropout(embeddings)

        # Layer 2
        support = torch.mm(embeddings, self.W2)
        embeddings = (
            torch.sparse.mm(adj, support) if self.is_sparse else torch.mm(adj, support)
        )

        return embeddings


class Readout(nn.Module):
    """
    This module learns a single graph level representation for a molecule given GNN generated node embeddings
    """

    def __init__(self, attr_dim, embedding_dim, hidden_dim, output_dim, num_cats):
        super(Readout, self).__init__()
        self.attr_dim = attr_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_cats = num_cats

        self.layer1 = nn.Linear(attr_dim + embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.output = nn.Linear(output_dim, num_cats)
        self.act = nn.ReLU()

    def forward(self, node_features, node_embeddings):
        combined_rep = torch.cat(
            (node_features, node_embeddings), dim=1
        )  # Concat initial node attributed with embeddings from sage
        hidden_rep = self.act(self.layer1(combined_rep))
        graph_rep = self.act(
            self.layer2(hidden_rep)
        )  # Generate final graph level embedding

        logits = torch.mean(
            self.output(graph_rep), dim=0
        )  # Generated logits for multilabel classification

        return logits


class GcnMoleculeNet(nn.Module):
    """
    Network that consolidates GCN + Readout into a single nn.Module
    """

    def __init__(
        self,
        feat_dim,
        hidden_dim,
        node_embedding_dim,
        dropout,
        readout_hidden_dim,
        graph_embedding_dim,
        num_categories,
        sparse_adj=False,
    ):
        super(GcnMoleculeNet, self).__init__()
        self.gcn = GCN(
            feat_dim, hidden_dim, node_embedding_dim, dropout, is_sparse=sparse_adj
        )
        self.readout = Readout(
            feat_dim,
            node_embedding_dim,
            readout_hidden_dim,
            graph_embedding_dim,
            num_categories,
        )

    def forward(self, adj_matrix, feature_matrix):
        node_embeddings = self.gcn(feature_matrix, adj_matrix)
        logits = self.readout(feature_matrix, node_embeddings)
        return logits
