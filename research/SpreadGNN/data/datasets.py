import logging

import numpy as np
import scipy
import pickle
import gzip
import os
from tqdm import tqdm

import torch.utils.data as data


# From GTTF, need to cite once paper is officially accepted to ICLR 2021
class CompactAdjacency:
    def __init__(self, adj, precomputed=None, subset=None):
        """
        Constructs a CompactAdjacency object.

        Args:
            adj: scipy sparse matrix containing the full adjacency.
            precomputed: If given, must be a tuple (compact_adj, degrees).
                In this case, adj must be None. If supplied, subset will be ignored.
            subset: Optional set of node indices to consider in the adjacency matrix.

        Note:
            This constructor initializes a CompactAdjacency object based on the provided arguments.
            If 'precomputed' is provided, 'adj' and 'subset' will be ignored.

        Raises:
            ValueError: If both 'adj' and 'precomputed' are set.
        """
        if adj is None:
            return

        if precomputed:
            if adj is not None:
                raise ValueError("Both adj and precomputed are set.")
            if subset is not None:
                logging.info(
                    "WARNING: subset is provided. It is ignored, since precomputed is supplied."
                )
            self.compact_adj, self.degrees = precomputed
            self.num_nodes = len(self.degrees)
        else:
            self.adj = adj
            self.num_nodes = (
                len(self.adj) if isinstance(self.adj, dict) else self.adj.shape[0]
            )
            self.compact_adj = scipy.sparse.dok_matrix(
                (self.num_nodes, self.num_nodes), dtype="int32"
            )
            self.degrees = np.zeros(shape=[self.num_nodes], dtype="int32")
            self.node_set = set(subset) if subset is not None else None

            for v in range(self.num_nodes):
                if isinstance(self.adj, dict) and self.node_set is not None:
                    connection_ids = np.array(
                        list(self.adj[v].intersection(self.node_set))
                    )
                elif isinstance(self.adj, dict) and self.node_set is None:
                    connection_ids = np.array(list(self.adj[v]))
                else:
                    connection_ids = self.adj[v].nonzero()[1]

                self.degrees[v] = len(connection_ids)
                self.compact_adj[
                    v, np.arange(len(connection_ids), dtype="int32")
                ] = connection_ids

        self.compact_adj = self.compact_adj.tocsr()

    @staticmethod
    def from_file(filename):
        instance = CompactAdjacency(None, None)
        data = pickle.load(gzip.open(filename, "rb"))
        instance.compact_adj = data["compact_adj"]
        instance.adj = data["adj"]
        instance.degrees = data["degrees"] if "degrees" in data else data["lengths"]
        instance.num_nodes = data["num_nodes"]
        return instance

    @staticmethod
    def from_directory(directory):
        instance = CompactAdjacency(None, None)
        instance.degrees = np.load(os.path.join(directory, "degrees.npy"))
        instance.compact_adj = scipy.sparse.load_npz(
            os.path.join(directory, "cadj.npz")
        )
        logging.info("\n\ncompact_adj.py from_directory\n\n")
        # Make adj from cadj and save to adj.npz
        import IPython

        IPython.embed()
        instance.adj = scipy.sparse.load_npz(os.path.join(directory, "adj.npz"))
        instance.num_nodes = instance.adj.shape[0]
        return instance

    def save(self, filename):
        with gzip.open(filename, "wb") as fout:
            pickle.dump(
                {
                    "compact_adj": self.compact_adj,
                    "adj": self.adj,
                    "degrees": self.degrees,
                    "num_nodes": self.num_nodes,
                },
                fout,
            )

    def neighbors_of(self, node):
        neighbors = self.compact_adj[node, : self.degrees[node]].todense()
        return np.array(neighbors)[0]


class MoleculesDataset(data.Dataset):
    def __init__(
        self,
        adj_matrices,
        feature_matrices,
        labels,
        path,
        compact=True,
        fanouts=[2, 2],
        split="train",
    ):
        """
        Constructs a dataset for molecules with adjacency matrices, feature matrices, and labels.

        Args:
            adj_matrices (list): A list of adjacency matrices.
            feature_matrices (list): A list of feature matrices.
            labels (list): A list of labels.
            path (str): The path to the directory containing data files.
            compact (bool, optional): Whether to use compact adjacency matrices. Defaults to True.
            fanouts (list, optional): A list of fanout values for each adjacency matrix. Defaults to [2, 2].
            split (str, optional): The dataset split ('train', 'val', or 'test'). Defaults to 'train'.

        Note:
            This constructor initializes a MoleculesDataset object based on the provided arguments.
            If 'compact' is set to True, it uses compact adjacency matrices.

        Raises:
            None
        """
        if compact:
            # filename = path + '/train_comp_adjs.pkl'
            # if split == 'val':
            #     filename = path + '/val_comp_adjs.pkl'
            # elif split == 'test':
            #     filename = path + '/test_comp_adjs.pkl'
            #
            # if os.path.isfile(filename):
            #     print('Loading saved compact adjacencies from disk!')
            #     with open(filename, 'rb') as f:
            #         self.adj_matrices = pickle.load(f)
            #
            # else:
            #     logging.info('Compacting adjacency matrices (GTTF)')
            #     self.adj_matrices = [CompactAdjacency(adj_matrix) for adj_matrix in tqdm(adj_matrices)]
            #     with open(filename, 'wb') as f:
            #         pickle.dump(self.adj_matrices, f)
            self.adj_matrices = [
                CompactAdjacency(adj_matrix) for adj_matrix in tqdm(adj_matrices)
            ]

        else:
            self.adj_matrices = adj_matrices

        self.feature_matrices = feature_matrices
        self.labels = labels
        self.fanouts = [fanouts] * len(adj_matrices)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the following elements:
                - adj_matrix: The adjacency matrix.
                - feature_matrix: The feature matrix.
                - label: The label.
                - fanouts: The list of fanout values.
        """
        return (
            self.adj_matrices[index],
            self.feature_matrices[index],
            self.labels[index],
            self.fanouts[index],
        )

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Args:
            None

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.adj_matrices)
