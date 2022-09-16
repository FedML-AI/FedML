from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any


class BaseContributionAssessor(ABC):

    @abstractmethod
    def run(
        self,
        num_client_for_this_round: int,
        model_list_from_client_update: List[Dict],
        model_aggregated: Dict,
        model_last_round: Dict,
        acc_on_aggregated_model: float,
        val_dataloader: Any,
        validation_func: Callable[[Dict, Any, Any], float],
        device,
    ) -> List[float]:
        pass
