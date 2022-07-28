import abc


class FedMLExecutor(abc.ABC):
    def __init__(self, id, neighbor_id_list):
        self.id = id
        self.neighbor_id_list = neighbor_id_list
        self.loop_times = 1
        self.params = None
        self.context = None

    def get_context(self):
        return self.context

    def set_context(self, context):
        self.context = context

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

    def set_loop_times(self, loop_times):
        self.loop_times = loop_times

    def get_loop_times(self):
        return self.loop_times

    def set_id(self, id):
        self.id = id

    def set_neighbor_id_list(self, neighbor_id_list):
        self.neighbor_id_list = neighbor_id_list

    def get_id(self):
        return self.id

    def get_neighbor_id_list(self):
        return self.neighbor_id_list
