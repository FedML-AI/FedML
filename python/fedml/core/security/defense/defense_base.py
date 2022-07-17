from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class BaseDefenseMethod(ABC):
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
