import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    """
    A single Graph Attention Layer (GAT) module.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        dropout (float): Dropout probability for attention coefficients.
        alpha (float): LeakyReLU slope parameter.
        concat (bool): Whether to concatenate the multi-head results or not.

    Attributes:
        W (nn.Parameter): Learnable weight matrix.
        a (nn.Parameter): Learnable attention parameter matrix.
        leakyrelu (nn.LeakyReLU): LeakyReLU activation with slope alpha.

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
        """
        Forward pass for the GAT layer.

        Args:
            h (torch.Tensor): Input feature tensor.
            adj (torch.Tensor): Adjacency matrix.

        Returns:
            torch.Tensor: Output feature tensor.

        """
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
        """
        String representation of the GAT layer.

        Returns:
            str: A string representing the layer.

        """
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GAT(nn.Module):
    """
    Graph Attention Network (GAT) model for node classification.

    Args:
        feat_dim (int): Number of input features.
        hidden_dim1 (int): Number of hidden units in the first GAT layer.
        hidden_dim2 (int): Number of hidden units in the second GAT layer.
        dropout (float): Dropout probability for attention coefficients.
        alpha (float): LeakyReLU slope parameter.
        nheads (int): Number of attention heads.

    Attributes:
        dropout (float): Dropout probability.
        attentions (nn.ModuleList): List of GAT layers with multiple heads.
        out_att (GraphAttentionLayer): Final GAT layer.

    """ 
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
        """
        Forward pass for the GAT model.

        Args:
            x (torch.Tensor): Input feature tensor.
            adj (torch.Tensor): Adjacency matrix.

        Returns:
            torch.Tensor: Node embeddings.

        """
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        embeddings = F.elu(self.out_att(x, adj))  # Node embeddings

        return embeddings


class Readout(nn.Module):
    """
    This module learns a single graph-level representation for a molecule given GNN-generated node embeddings.

    Args:
        attr_dim (int): Dimension of node attributes.
        embedding_dim (int): Dimension of node embeddings.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output layer.
        num_cats (int): Number of categories for classification.

    Attributes:
        attr_dim (int): Dimension of node attributes.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output layer.
        num_cats (int): Number of categories for classification.
        layer1 (nn.Linear): First linear layer.
        layer2 (nn.Linear): Second linear layer.
        output (nn.Linear): Output layer.
        act (nn.ReLU): ReLU activation function.

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
        """
        Forward pass for the Readout module.

        Args:
            node_features (torch.Tensor): Node attributes.
            node_embeddings (torch.Tensor): Node embeddings.

        Returns:
            torch.Tensor: Logits for multilabel classification.

        """
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
    Neural network that combines GAT (Graph Attention Network) and Readout into a single module for molecular data.

    Args:
        feat_dim (int): Dimension of input node features.
        gat_hidden_dim1 (int): Dimension of the hidden layer in the GAT model.
        node_embedding_dim (int): Dimension of node embeddings.
        gat_dropout (float): Dropout probability for GAT layers.
        gat_alpha (float): LeakyReLU slope parameter for GAT.
        gat_nheads (int): Number of attention heads in GAT.
        readout_hidden_dim (int): Dimension of the hidden layer in the Readout module.
        graph_embedding_dim (int): Dimension of the graph-level embedding.
        num_categories (int): Number of categories for classification.

    Attributes:
        gat (GAT): GAT (Graph Attention Network) module.
        readout (Readout): Readout module for graph-level representation.

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
