from abc import ABC, abstractmethod
from itertools import chain, combinations
from typing import Callable, List, Dict, Any

"""
# discussion about integration

# should we return dict for contributions?
# is the model dict?
# we need to have retrain for the exact SV/
"""

class BaseContributionAssessor(ABC):
    @abstractmethod
    def run(
        self,
        num_client_for_this_round: int,
        client_index_for_this_round: List,  # this is the indices of the participating users in that iteration
        aggregation_func: Callable,
        local_weights_from_clients: List[Dict],  # TO DO dict: [id]
        acc_on_last_round: float,
        acc_on_aggregated_model: float,
        val_dataloader: Any,
        validation_func: Callable[[Dict, Any, Any], float],
        device,
    ):  # -> List[float]:
        pass

    @abstractmethod
    def get_final_contribution_assignment(self) -> dict:
        pass

    @staticmethod
    def get_aggregated_model_with_client_subset(
        args, aggregation_func, local_weights_from_clients, client_subset_list
    ):
        """
        Constructs an aggregate model from local updates of the users.
        """
        local_weights_from_subset = {
            client_index: local_weights_from_clients[client_index] for client_index in client_subset_list
        }
        return aggregation_func(args, local_weights_from_subset)

    @staticmethod
    def generate_power_set(input_iterable):
        """
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        """
        s = list(input_iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
