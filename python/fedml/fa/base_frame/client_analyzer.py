import logging
from abc import ABC, abstractmethod


class FAClientAnalyzer(ABC):
    def __init__(self, args):
        self.client_submission = 0
        self.id = 0
        self.args = args
        self.local_train_dataset = None
        self.server_data = None
        self.init_msg = None

    def set_init_msg(self, init_msg):
        pass

    def get_init_msg(self):
        pass

    def set_id(self, analyzer_id):
        self.id = analyzer_id

    def get_client_submission(self):
        return self.client_submission

    def set_client_submission(self, client_submission):
        self.client_submission = client_submission

    def get_server_data(self):
        return self.server_data

    def set_server_data(self, server_data):
        self.server_data = server_data

    @abstractmethod
    def local_analyze(self, train_data, args):
        pass

    def update_dataset(self, local_train_dataset, local_sample_number):
        self.local_train_dataset = local_train_dataset
        self.local_sample_number = local_sample_number
