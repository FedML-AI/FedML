from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Callable


class ServerAggregator(ABC):
    """Abstract base class for federated learning trainer."""

    def __init__(self, model, args=None):
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

    @abstractmethod
    def on_before_aggregation(
        self,
        raw_client_model_or_grad_list: List[Tuple[float, Dict]]
    ) -> List[Tuple[float, Dict]]:
        pass

    @abstractmethod
    def aggregate(
        self,
        raw_client_model_or_grad_list: List[Tuple[float, Dict]],
        base_aggregation_func: Callable = None
    ) -> Dict:
        pass

    @abstractmethod
    def on_after_aggregation(
        self,
        aggregated_model_or_grad: Dict
    ) -> Dict:
        pass

    @abstractmethod
    def eval(self, eval_data, device, args=None):
        pass

    @abstractmethod
    def test(self, test_data, device, args=None):
        pass
