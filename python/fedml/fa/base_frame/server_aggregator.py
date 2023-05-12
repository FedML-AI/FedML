from abc import ABC
from typing import List, Tuple, Any


class FAServerAggregator(ABC):
    def __init__(self, args):
        self.id = 0
        self.args = args
        self.eval_data = None
        self.server_data = None
        self.init_msg = None

    def get_init_msg(self):
        # return self.init_msg
        pass

    def set_init_msg(self, init_msg):
        # self.init_msg = init_msg
        pass

    def set_id(self, aggregator_id):
        self.id = aggregator_id

    def get_server_data(self):
        return self.server_data

    def set_server_data(self, server_data):
        self.server_data = server_data

    def aggregate(self, local_submissions: List[Tuple[float, Any]]):
        pass
