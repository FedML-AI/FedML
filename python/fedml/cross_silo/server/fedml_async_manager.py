import logging


class FedMLAsyncManager:
    _async_instance = None

    @staticmethod
    def get_instance():
        if FedMLAsyncManager._async_instance is None:
            FedMLAsyncManager._async_instance = FedMLAsyncManager()
        return FedMLAsyncManager._async_instance

    def __init__(self):
        self.is_async = False
        self.batch_size = 0
        self.local_model_counter = 0
        self.max_staleness = 0
        self.stalness_factor = 0

    def init(self, config):
        self.is_async = False
        if hasattr(config, "is_async") and isinstance(config.is_async, bool) and config.is_async:
            self.is_async = True
            if hasattr(config, "batch_size") and isinstance(config.batch_size, int) and config.batch_size > 0:
                if config.batch_size > config.client_num_per_round:
                    self.batch_size = config.client_num_per_round
                else:
                    self.batch_size = config.batch_size
            else:
                self.batch_size = config.client_num_per_round
            if hasattr(config, "max_staleness") and isinstance(config.max_staleness, int) and config.max_staleness > 0:
                self.max_staleness = config.max_staleness
            else:
                self.max_staleness = 3

    def is_enabled(self):
        return self.is_async

    def get_stalness_factor(self):
        return self.stalness_factor

    @staticmethod
    def find_staleness(current_round_idx, local_model_round_idx):
        return current_round_idx - int(local_model_round_idx)

    def set_staleness_factor(self, staleness):
        self.stalness_factor = 1/(1+staleness) ** 0.5

    def add_local_model_counter(self):
        self.local_model_counter += 1

    def get_local_model_counter(self):
        return self.local_model_counter

    def is_to_aggregate(self):
        if self.local_model_counter == self.batch_size:
            self.local_model_counter = 0
            return True
        return False
