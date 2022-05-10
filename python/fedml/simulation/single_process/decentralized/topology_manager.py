import networkx as nx
import numpy as np


class TopologyManager:
    def __init__(
        self, n, b_symmetric, undirected_neighbor_num=5, out_directed_neighbor=5
    ):
        self.n = n
        self.b_symmetric = b_symmetric
        self.undirected_neighbor_num = undirected_neighbor_num
        self.out_directed_neighbor = out_directed_neighbor
        self.topology_symmetric = []
        self.topology_asymmetric = []
        self.b_fully_connected = False
        if self.undirected_neighbor_num >= self.n - 1 and self.b_symmetric:
            self.b_fully_connected = True

    def generate_topology(self):
        if self.b_fully_connected:
            self.__fully_connected()
            return

        if self.b_symmetric:
            self.__randomly_pick_neighbors_symmetric()
        else:
            self.__randomly_pick_neighbors_asymmetric()

    def get_symmetric_neighbor_list(self, client_idx):
        if client_idx >= self.n:
            return []
        return self.topology_symmetric[client_idx]

    def get_asymmetric_neighbor_list(self, client_idx):
        if client_idx >= self.n:
            return []
        return self.topology_asymmetric[client_idx]

    def __randomly_pick_neighbors_symmetric(self):
        # first generate a ring topology
        topology_ring = np.array(
            nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n, 2, 0)), dtype=np.float32
        )
        # print(topology_ring)

        # randomly add some links for each node (symmetric)
        k = int(self.undirected_neighbor_num)
        # print("undirected_neighbor_num = " + str(k))
        topology_random_link = np.array(
            nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n, k, 0)), dtype=np.float32
        )
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

        self.topology_symmetric = topology_symmetric

    def __randomly_pick_neighbors_asymmetric(self):
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
        self.topology_asymmetric = topology_ring

    def __fully_connected(self):
        topology_fully_connected = np.array(
            nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n, self.n - 1, 0)),
            dtype=np.float32,
        )
        for i in range(self.n):
            for j in range(self.n):
                if topology_fully_connected[i][j] != 1:
                    topology_fully_connected[i][j] = 1

        for i in range(self.n):
            row_len_i = 0
            for j in range(self.n):
                if topology_fully_connected[i][j] == 1:
                    row_len_i += 1
            topology_fully_connected[i] = topology_fully_connected[i] / row_len_i

        self.topology_symmetric = topology_fully_connected


if __name__ == "__main__":
    tpmgr = TopologyManager(16, False, 4, 4)
    tpmgr.generate_topology()
