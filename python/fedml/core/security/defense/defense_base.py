from abc import ABC, abstractmethod
from typing import List


class BaseDefenseMethod(ABC):
    @abstractmethod
    def defend(self, client_grad_list: List) -> List:
        """
        client_grad_list is a list, each item is (num_samples, gradients)
        """
        pass