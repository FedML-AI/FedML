from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class BaseDefenseMethod(ABC):
    @abstractmethod
    def defend(
        self, client_grad_list: List[Tuple[int, Dict]]
    ) -> List[Tuple[int, Dict]]:
        """
        args:
            client_grad_list is a list, each item is (num_samples, gradients)
        return:
            Note: the data type of the return variable should be the same as the input
        """
        pass
