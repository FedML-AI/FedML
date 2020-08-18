import networkx as nx
import numpy as np

from fedml_core.distributed.topology.base_topology_manager import BaseTopologyManager


class SymmetricTopologyManager(BaseTopologyManager):
    """
    The topology definition is determined by this initialization method.

    Arguments:
        n (int): number of nodes in the topology.
        neighbor_num (int): number of neighbors for each node
    """
    def __init__(self, n, neighbor_num=2):
        self.n = n
        self.neighbor_num = neighbor_num
        self.topology = []

    def generate_topology(self):
        # first generate a ring topology
        topology_ring = np.array(nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n, 2, 0)), dtype=np.float32)
        # print(topology_ring)

        # randomly add some links for each node (symmetric)
        k = int(self.neighbor_num)
        # print("undirected_neighbor_num = " + str(k))
        topology_random_link = np.array(nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n, k, 0)), dtype=np.float32)
        # print("randomly add some links for each node (symmetric): ")
        # print(topology_random_link)

        # generate symmetric topology
        topology_symmetric = topology_ring.copy()
        for i in range(self.n):
            for j in range(self.n):
                if topology_symmetric[i][j] == 0 and topology_random_link[i][j] == 1:
                    topology_symmetric[i][j] = topology_random_link[i][j]
        np.fill_diagonal(topology_symmetric, 1)
        # print("symmetric topology:")
        # print(topology_symmetric)

        for i in range(self.n):
            row_len_i = 0
            for j in range(self.n):
                if topology_symmetric[i][j] == 1:
                    row_len_i += 1
            topology_symmetric[i] = topology_symmetric[i] / row_len_i
        # print("weighted symmetric confusion matrix:")
        # print(topology_symmetric)

        self.topology = topology_symmetric

    def get_in_neighbor_list(self, node_index):
        if node_index >= self.n:
            return []
        return self.topology[node_index]

    def get_out_neighbor_list(self, node_index):
        if node_index >= self.n:
            return []
        return self.topology[node_index]


if __name__ == "__main__":
    tpmgr = SymmetricTopologyManager(6, 2)
    tpmgr.generate_topology()
    print(tpmgr.topology)

    # get the out neighbor list for node 1
    out_neighbor_list = tpmgr.get_out_neighbor_list(1)
    print(out_neighbor_list)