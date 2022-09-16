from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseContributionAssessor(ABC):
    @abstractmethod
    def run(
        self,
        num_client_for_this_round: int,
        model_list_from_client_update: List[Dict],
        model_aggregated: Dict,
        model_last_round: Dict,
        acc_on_aggregated_model: float,
        val_dataset: Any,
    ) -> List[float]:
        pass
