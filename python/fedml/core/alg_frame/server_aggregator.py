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
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_data, device, args=None):
        pass

    @abstractmethod
    def test(self, test_data, device, args=None):
        pass

    @abstractmethod
    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        pass
