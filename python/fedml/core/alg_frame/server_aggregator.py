from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from fedml.ml.aggregator.agg_operator import FedMLAggOperator


class ServerAggregator(ABC):
    """Abstract base class for federated learning trainer."""

    def __init__(self, model, args):
        self.model = model
        self.id = 0
        self.args = args

    def set_id(self, aggregator_id):
        self.id = aggregator_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    def on_before_aggregation(
        self, raw_client_model_or_grad_list: List[Tuple[float, Dict]]
    ) -> List[Tuple[float, Dict]]:
        pass

    def aggregate(self, raw_client_model_or_grad_list: List[Tuple[float, Dict]]) -> Dict:
        return FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)

    def on_after_aggregation(self, aggregated_model_or_grad: Dict) -> Dict:
        pass

    @abstractmethod
    def test(self, test_data, device, args):
        pass

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        pass
