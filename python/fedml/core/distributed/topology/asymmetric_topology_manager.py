import networkx as nx
import numpy as np

from .base_topology_manager import BaseTopologyManager


class AsymmetricTopologyManager(BaseTopologyManager):
    """
    The topology definition is determined by this initialization method.

    Arguments:
        n (int): number of nodes in the topology.
        undirected_neighbor_num (int): number of undirected (symmetric) neighbors for each node
        out_directed_neighbor (int): number of out (asymmetric) neighbors for each node
    """

    def __init__(self, n, undirected_neighbor_num=3, out_directed_neighbor=3):
        self.n = n
        self.undirected_neighbor_num = undirected_neighbor_num
        self.out_directed_neighbor = out_directed_neighbor
        self.topology = []

    def generate_topology(self):
        # randomly add some links for each node (symmetric)
        k = self.undirected_neighbor_num
        # print("neighbors = " + str(k))
        topology_random_link = np.array(
            nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n, k, 0)), dtype=np.float32
        )
        # print("randomly add some links for each node (symmetric): ")
        # print(topology_random_link)

        # first generate a ring topology
        topology_ring = np.array(
            nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n, 2, 0)), dtype=np.float32
        )

        for i in range(self.n):
            for j in range(self.n):
                if topology_ring[i][j] == 0 and topology_random_link[i][j] == 1:
                    topology_ring[i][j] = topology_random_link[i][j]

        np.fill_diagonal(topology_ring, 1)

        # k_d = self.out_directed_neighbor
        # Directed graph
        # Undirected graph
        # randomly delete some links
        out_link_set = set()
        for i in range(self.n):
            len_row_zero = 0
            for j in range(self.n):
                if topology_ring[i][j] == 0:
                    len_row_zero += 1
            random_selection = np.random.randint(2, size=len_row_zero)
            # print(random_selection)
            index_of_zero = 0
            for j in range(self.n):
                out_link = j * self.n + i
                if topology_ring[i][j] == 0:
                    if (
                        random_selection[index_of_zero] == 1
                        and out_link not in out_link_set
                    ):
                        topology_ring[i][j] = 1
                        out_link_set.add(i * self.n + j)
                    index_of_zero += 1

        # print("asymmetric topology:")
        # print(topology_ring)

        for i in range(self.n):
            row_len_i = 0
            for j in range(self.n):
                if topology_ring[i][j] == 1:
                    row_len_i += 1
            topology_ring[i] = topology_ring[i] / row_len_i

        # print("weighted asymmetric confusion matrix:")
        # print(topology_ring)
        self.topology = topology_ring

    def get_in_neighbor_weights(self, node_index):
        if node_index >= self.n:
            return []
        in_neighbor_weights = []
        for row_idx in range(len(self.topology)):
            in_neighbor_weights.append(self.topology[row_idx][node_index])
        return in_neighbor_weights

    def get_out_neighbor_weights(self, node_index):
        if node_index >= self.n:
            return []
        return self.topology[node_index]

    def get_in_neighbor_idx_list(self, node_index):
        neighbor_in_idx_list = []
        neighbor_weights = self.get_in_neighbor_weights(node_index)
        for idx, neighbor_w in enumerate(neighbor_weights):
            if neighbor_w > 0 and node_index != idx:
                neighbor_in_idx_list.append(idx)
        return neighbor_in_idx_list

    def get_out_neighbor_idx_list(self, node_index):
        neighbor_out_idx_list = []
        neighbor_weights = self.get_out_neighbor_weights(node_index)
        for idx, neighbor_w in enumerate(neighbor_weights):
            if neighbor_w > 0 and node_index != idx:
                neighbor_out_idx_list.append(idx)
        return neighbor_out_idx_list


if __name__ == "__main__":
    # generate a asymmetric topology
    tpmgr = AsymmetricTopologyManager(8, 4, 2)
    tpmgr.generate_topology()
    print("tpmgr.topology = " + str(tpmgr.topology))

    # get the OUT neighbor weights for node 1
    out_neighbor_weights = tpmgr.get_out_neighbor_weights(1)
    print("out_neighbor_weights = " + str(out_neighbor_weights))

    # get the OUT neighbor index list for node 1
    out_neighbor_idx_list = tpmgr.get_out_neighbor_idx_list(1)
    print("out_neighbor_idx_list = " + str(out_neighbor_idx_list))

    # get the IN neighbor weights for node 1
    in_neighbor_weights = tpmgr.get_in_neighbor_weights(1)
    print("in_neighbor_weights = " + str(in_neighbor_weights))

    # get the IN neighbor index list for node 1
    in_neighbor_idx_list = tpmgr.get_in_neighbor_idx_list(1)
    print("in_neighbor_idx_list = " + str(in_neighbor_idx_list))
