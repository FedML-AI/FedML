import abc


class BaseTopologyManager(abc.ABC):

    @abc.abstractmethod
    def generate_topology(self):
        pass

    @abc.abstractmethod
    def get_in_neighbor_list(self, node_index):
        pass

    @abc.abstractmethod
    def get_out_neighbor_list(self, node_index):
        pass
