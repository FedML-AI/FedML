import torch
import torch.nn as nn


class GraphSage(nn.Module):
    """
    GraphSAGE model (https://arxiv.org/abs/1706.02216) to learn the role of atoms in the molecules inductively.
    Transforms input features into a fixed length embedding in a vector space. The embedding captures the role.
    """
    """
    GraphSAGE model to learn the role of atoms in molecules inductively.

    GraphSAGE (Graph Sample and Aggregated) transforms input features into a fixed-length embedding in a vector space. 
    The resulting embedding captures the role of atoms in molecules.

    Args:
        feat_dim (int): Dimension of input node features.
        hidden_dim1 (int): Dimension of the first hidden layer.
        hidden_dim2 (int): Dimension of the second hidden layer.
        dropout (float): Dropout probability.

    Attributes:
        feat_dim (int): Dimension of input node features.
        hidden_dim1 (int): Dimension of the first hidden layer.
        hidden_dim2 (int): Dimension of the second hidden layer.
        layer1 (nn.Linear): First linear layer for feature transformation.
        layer2 (nn.Linear): Second linear layer for feature transformation.
        relu (nn.ReLU): ReLU activation function.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GraphSage, self).__init__()

        self.feat_dim = feat_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.layer1 = nn.Linear(2 * feat_dim, hidden_dim1, bias=False)
        self.layer2 = nn.Linear(2 * hidden_dim1, hidden_dim2, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, forest, feature_matrix):
        feat_0 = feature_matrix[forest[0]]  # Of shape torch.Size([|B|, feat_dim])
        feat_1 = feature_matrix[
            forest[1]
        ]  # Of shape torch.size(|B|, fanouts[0], feat_dim)

        # Depth 1
        x = feature_matrix[forest[1]].mean(dim=1)  # Of shape torch.size(|B|, feat_dim)
        feat_0 = torch.cat((feat_0, x), dim=1)  # Of shape torch.size(|B|, 2 * feat_dim)
        feat_0 = self.relu(self.layer1(feat_0))  # Of shape torch.size(|B|, hidden_dim1)
        feat_0 = self.dropout(feat_0)

        # Depth 2
        x = feature_matrix[forest[2]].mean(
            dim=1
        )  # Of shape torch.size(|B|*fanouts[0], feat_dim)
        feat_1 = torch.cat(
            (feat_1.reshape(-1, self.feat_dim), x), dim=1
        )  # Of shape torch.size(|B|*fanouts[0], 2 * feat_dim)
        feat_1 = self.relu(
            self.layer1(feat_1)
        )  # Of shape torch.size(|B|*fanouts[0], hidden_dim1)
        feat_1 = self.dropout(feat_1)

        # Combine
        feat_1 = feat_1.reshape(forest[0].shape[0], -1, self.hidden_dim1).mean(
            dim=1
        )  # Of shape torch.size([|B|, hidden_dim_1])
        combined = torch.cat(
            (feat_0, feat_1), dim=1
        )  # Of shape torch.Size(|B|, 2 * hidden_dim1)
        embeddings = self.relu(
            self.layer2(combined)
        )  # Of shape torch.Size(|B|, hidden_dim2)

        return embeddings


class Readout(nn.Module):
    """
    This module learns a single graph-level representation for a molecule using GraphSAGE-generated embeddings.

    The Readout module combines node-level features and GraphSAGE-generated embeddings to produce a graph-level representation of a molecule. This representation can be used for various downstream tasks, such as multi-label classification.

    Args:
        attr_dim (int): Dimension of initial node attributes.
        embedding_dim (int): Dimension of GraphSAGE-generated node embeddings.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output layer.
        num_cats (int): Number of categories for classification.

    Attributes:
        attr_dim (int): Dimension of initial node attributes.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output layer.
        num_cats (int): Number of categories for classification.
        layer1 (nn.Linear): First linear layer for feature transformation.
        layer2 (nn.Linear): Second linear layer for feature transformation.
        output (nn.Linear): Output linear layer.
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


class SageMoleculeNet(nn.Module):
    """
    Network that combines Sage (GraphSAGE) and Readout into a single neural network module.

    The SageMoleculeNet module integrates GraphSAGE for node-level embedding generation and a Readout module for graph-level representation of molecules. It is designed for tasks such as multi-label classification on molecular graphs.

    Args:
        feat_dim (int): Dimension of node features.
        sage_hidden_dim1 (int): Dimension of the first hidden layer in GraphSAGE.
        node_embedding_dim (int): Dimension of node embeddings generated by GraphSAGE.
        sage_dropout (float): Dropout rate for GraphSAGE.
        readout_hidden_dim (int): Dimension of the hidden layer in the Readout module.
        graph_embedding_dim (int): Dimension of the graph-level embeddings.
        num_categories (int): Number of categories for classification.

    Attributes:
        sage (GraphSage): GraphSAGE module for node-level embedding generation.
        readout (Readout): Readout module for generating graph-level representations.

    """

    def __init__(
        self,
        feat_dim,
        sage_hidden_dim1,
        node_embedding_dim,
        sage_dropout,
        readout_hidden_dim,
        graph_embedding_dim,
        num_categories,
    ):
        super(SageMoleculeNet, self).__init__()
        self.sage = GraphSage(
            feat_dim, sage_hidden_dim1, node_embedding_dim, sage_dropout
        )
        self.readout = Readout(
            feat_dim,
            node_embedding_dim,
            readout_hidden_dim,
            graph_embedding_dim,
            num_categories,
        )

    def forward(self, forest, feature_matrix):
        node_embeddings = self.sage(forest, feature_matrix)
        logits = self.readout(feature_matrix, node_embeddings)
        return logits
