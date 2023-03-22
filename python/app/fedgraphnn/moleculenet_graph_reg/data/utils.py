import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler


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
