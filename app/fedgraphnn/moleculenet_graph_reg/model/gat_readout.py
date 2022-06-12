import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(
            h, self.W
        )  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1
        )

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GAT(nn.Module):
    def __init__(self, feat_dim, hidden_dim1, hidden_dim2, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList(
            [
                GraphAttentionLayer(
                    feat_dim, hidden_dim1, dropout=dropout, alpha=alpha, concat=True
                )
                for _ in range(nheads)
            ]
        )
        self.out_att = GraphAttentionLayer(
            hidden_dim1 * nheads,
            hidden_dim2,
            dropout=dropout,
            alpha=alpha,
            concat=False,
        )

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        embeddings = F.elu(self.out_att(x, adj))  # Node embeddings

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


class GatMoleculeNet(nn.Module):
    """
    Network that consolidates GAT + Readout into a single nn.Module
    """

    def __init__(
        self,
        feat_dim,
        gat_hidden_dim1,
        node_embedding_dim,
        gat_dropout,
        gat_alpha,
        gat_nheads,
        readout_hidden_dim,
        graph_embedding_dim,
        num_categories,
    ):
        super(GatMoleculeNet, self).__init__()
        self.gat = GAT(
            feat_dim,
            gat_hidden_dim1,
            node_embedding_dim,
            gat_dropout,
            gat_alpha,
            gat_nheads,
        )
        self.readout = Readout(
            feat_dim,
            node_embedding_dim,
            readout_hidden_dim,
            graph_embedding_dim,
            num_categories,
        )

    def forward(self, adj_matrix, feature_matrix):
        node_embeddings = self.gat(feature_matrix, adj_matrix)
        logits = self.readout(feature_matrix, node_embeddings)
        return logits
