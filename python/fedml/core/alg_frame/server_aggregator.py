from abc import ABC, abstractmethod


class ServerAggregator(ABC):
    """Abstract base class for federated learning trainer."""

    def __init__(self, model, args=None):
        self.model = model
        self.id = 0
        self.args = args

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def init_global_model_params(self):
        pass

    @abstractmethod
    def set_global_model_params(self, model_parameters):
        pass

    @abstractmethod
    def aggregate(self):
        pass

    @abstractmethod
    def client_selection(self):
        pass

    @abstractmethod
    def eval(self):
        pass
