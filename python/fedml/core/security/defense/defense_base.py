from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Callable


class BaseDefenseMethod(ABC):
    @abstractmethod
    def run(
        self,
        base_aggregation_func: Callable,
        raw_client_grad_list: List[Tuple[int, Dict]],
        extra_auxiliary_info: Any = None
    ) -> Dict:
        """
        args:
            base_aggregation_func: this can be aggregation function in FedAvg, FedOPT, or any other machine learning optimizer.
            client_grad_list: client_grad_list is a list, each item is (sample_num, gradients)
            extra_auxiliary_info: for methods which need extra info (e.g., data, previous model/gradient),
                                please use this variable.
        return:
            Note: the data type of the return variable should be the same as the input
        """
        pass


    @abstractmethod
    def defend(
        self, client_grad_list: List[Tuple[int, Dict]], global_w=None
    ) -> List[Tuple[int, Dict]]:
        """
        args:
            client_grad_list is a list, each item is (sample_num, gradients)
        return:
            Note: the data type of the return variable should be the same as the input
        """
        pass

    def robust_aggregate(
        self, client_grad_list: List[Tuple[int, Dict]], global_w=None
    ) -> Dict:
        pass

    def robustify_global_model(self, avg_params, previous_global_w=None) -> Dict:
        pass
