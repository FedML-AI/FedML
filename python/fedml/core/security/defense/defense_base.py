from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Callable


class BaseDefenseMethod(ABC):
    """
    Base class for defense methods in Federated Learning.

    Attributes:
        config: Configuration parameters for the defense method.
    """
    @abstractmethod
    def __init__(self, config):
        """
        Initialize the defense method with the specified configuration.

        Args:
            config: Configuration parameters for the defense method.
        """
        pass

    def defend_before_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            extra_auxiliary_info: Any = None,
    ) -> List[Tuple[float, OrderedDict]]:
        """
        Apply defense before model aggregation.

        Args:
            raw_client_grad_list (list): List of client gradients for the current round.
            extra_auxiliary_info: Additional information required for defense.

        Returns:
            list: List of defended client gradients.
        """
        pass

    def defend_on_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            base_aggregation_func: Callable = None,
            extra_auxiliary_info: Any = None,
    ) -> OrderedDict:
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

    def get_malicious_client_idxs(self):
        """
        Get the indices of potentially malicious clients.

        Returns:
            list: List of indices of potentially malicious clients.
        """
        return []
