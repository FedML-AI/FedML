class DecentralizedWorker(object):
    def __init__(self, worker_index, topology_manager):
        self.worker_index = worker_index
        self.in_neighbor_idx_list = topology_manager.get_in_neighbor_idx_list(
            self.worker_index
        )

        self.worker_result_dict = dict()
        self.flag_neighbor_result_received_dict = dict()
        for neighbor_idx in self.in_neighbor_idx_list:
            self.flag_neighbor_result_received_dict[neighbor_idx] = False

    def add_result(self, worker_index, updated_information):
        self.worker_result_dict[worker_index] = updated_information
        self.flag_neighbor_result_received_dict[worker_index] = True

    def check_whether_all_receive(self):
        for neighbor_idx in self.in_neighbor_idx_list:
            if not self.flag_neighbor_result_received_dict[neighbor_idx]:
                return False
        for neighbor_idx in self.in_neighbor_idx_list:
            self.flag_neighbor_result_received_dict[neighbor_idx] = False
        return True

    def train(self):
        self.add_result(self.worker_index, 0)
        return 0
