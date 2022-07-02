import logging
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_networkx, degree


def convert_to_nodeDegreeFeatures(graphs):
    collab_pickle_file = Path("./data/TUDataset/collab.pickle")
    if collab_pickle_file.exists():
        with open("./data/TUDataset/collab.pickle", "rb") as handle:
            new_graphs = pickle.load(handle)
            return new_graphs

    # print(graph.x)
    graph_infos = []
    maxdegree = 0
    logging.info("len(graphs) = {}".format(len(graphs)))
    for i, graph in enumerate(graphs):
        # logging.info("convert_to_nodeDegreeFeatures {}".format(i))
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append(
            (graph, g.degree, graph.num_nodes)
        )  # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tuple in enumerate(graph_infos):
        idx, x = tuple[0].edge_index[0], tuple[0].x
        deg = degree(idx, tuple[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tuple[0].clone()
        new_graph.__setitem__("x", deg)
        new_graphs.append(new_graph)

    with open("./data/TUDataset/collab.pickle", "wb") as handle:
        pickle.dump(new_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return new_graphs


def split_data(graphs, train=None, test=None, shuffle=True, seed=None):
    y = torch.cat([graph.y for graph in graphs])
    graphs_tv, graphs_test = train_test_split(
        graphs,
        train_size=train,
        test_size=test,
        stratify=y,
        shuffle=shuffle,
        random_state=seed,
    )
    return graphs_tv, graphs_test


def np_uniform_sample_next(compact_adj, tree, fanout):
    last_level = tree[-1]  # [batch, f^depth]
    batch_lengths = compact_adj.degrees[last_level]
    nodes = np.repeat(last_level, fanout, axis=1)
    batch_lengths = np.repeat(batch_lengths, fanout, axis=1)
    batch_next_neighbor_ids = np.random.uniform(
        size=batch_lengths.shape, low=0, high=1 - 1e-9
    )
    # Shape = (len(nodes), neighbors_per_node)
    batch_next_neighbor_ids = np.array(
        batch_next_neighbor_ids * batch_lengths, dtype=last_level.dtype
    )
    shape = batch_next_neighbor_ids.shape
    batch_next_neighbor_ids = np.array(
        compact_adj.compact_adj[nodes.reshape(-1), batch_next_neighbor_ids.reshape(-1)]
    ).reshape(shape)

    return batch_next_neighbor_ids


def np_traverse(
    compact_adj, seed_nodes, fanouts=(1,), sample_fn=np_uniform_sample_next
):
    if not isinstance(seed_nodes, np.ndarray):
        raise ValueError("Seed must a numpy array")

    if (
        len(seed_nodes.shape) > 2
        or len(seed_nodes.shape) < 1
        or not str(seed_nodes.dtype).startswith("int")
    ):
        raise ValueError("seed_nodes must be 1D or 2D int array")

    if len(seed_nodes.shape) == 1:
        seed_nodes = np.expand_dims(seed_nodes, 1)

    # Make walk-tree
    forest_array = [seed_nodes]
    for f in fanouts:
        next_level = sample_fn(compact_adj, forest_array, f)
        assert next_level.shape[1] == forest_array[-1].shape[1] * f

        forest_array.append(next_level)

    return forest_array


class WalkForestCollator(object):
    def __init__(self, normalize_features=False):
        self.normalize_features = normalize_features

    def __call__(self, molecule):
        comp_adj, feature_matrix, label, fanouts = molecule[0]
        node_ids = np.array(list(range(feature_matrix.shape[0])), dtype=np.int32)
        forest = np_traverse(comp_adj, node_ids, fanouts)
        torch_forest = [torch.from_numpy(forest[0]).flatten()]
        mask = np.where(np.isnan(label), 0.0, 1.0)
        label = np.where(np.isnan(label), 0.0, label)

        for i in range(len(forest) - 1):
            torch_forest.append(torch.from_numpy(forest[i + 1]).reshape(-1, fanouts[i]))

        if self.normalize_features:
            mx = sp.csr_matrix(feature_matrix)
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.0
            r_mat_inv = sp.diags(r_inv)
            normalized_feature_matrix = r_mat_inv.dot(mx)
            normalized_feature_matrix = np.array(normalized_feature_matrix.todense())
        else:
            scaler = StandardScaler()
            scaler.fit(feature_matrix)
            normalized_feature_matrix = scaler.transform(feature_matrix)

        return (
            torch_forest,
            torch.as_tensor(normalized_feature_matrix, dtype=torch.float32),
            torch.as_tensor(label, dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.float32),
        )


class DefaultCollator(object):
    def __init__(self, normalize_features=True, normalize_adj=True):
        self.normalize_features = normalize_features
        self.normalize_adj = normalize_adj

    def __call__(self, molecule):
        adj_matrix, feature_matrix, label, _ = molecule[0]
        mask = np.where(np.isnan(label), 0.0, 1.0)
        label = np.where(np.isnan(label), 0.0, label)

        if self.normalize_features:
            mx = sp.csr_matrix(feature_matrix)
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.0
            r_mat_inv = sp.diags(r_inv)
            normalized_feature_matrix = r_mat_inv.dot(mx)
            normalized_feature_matrix = np.array(normalized_feature_matrix.todense())
        else:
            scaler = StandardScaler()
            scaler.fit(feature_matrix)
            normalized_feature_matrix = scaler.transform(feature_matrix)

        if self.normalize_adj:
            rowsum = np.array(adj_matrix.sum(1))
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
            normalized_adj_matrix = (
                adj_matrix.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
            )
        else:
            normalized_adj_matrix = adj_matrix

        return (
            torch.as_tensor(
                np.array(normalized_adj_matrix.todense()), dtype=torch.float32
            ),
            torch.as_tensor(normalized_feature_matrix, dtype=torch.float32),
            torch.as_tensor(label, dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.float32),
        )
