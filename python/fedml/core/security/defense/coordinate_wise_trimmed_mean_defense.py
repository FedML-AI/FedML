from collections import OrderedDict
from typing import Callable, List, Tuple, Dict, Any
from .defense_base import BaseDefenseMethod
from ..common.utils import trimmed_mean

"""
added by Shanshan
Coordinate-wise Trimmed Mean from "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates".
This can be called at aggregate() of an Aggregator inplace of parameter averaging after \
model_list has been created
 """


class CoordinateWiseTrimmedMeanDefense(BaseDefenseMethod):
    """
    Coordinate-wise Trimmed Mean Defense for Federated Learning.

    Coordinate-wise Trimmed Mean Defense is a defense method for federated learning that computes the trimmed mean of
    gradients for each coordinate to mitigate the impact of Byzantine clients.

    Args:
        config: Configuration parameters for the defense, including 'beta' which represents the fraction of trimmed
        values; total trimmed values: client_num * beta * 2.

    Attributes:
        beta (float): The fraction of trimmed values, which determines the number of gradients to be trimmed on each side.
    """

    def __init__(self, config):
        """
        Initialize the CoordinateWiseTrimmedMeanDefense with the specified configuration.

        Args:
            config: Configuration parameters for the defense.
        """
        self.beta = config.beta

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        """
        Apply Coordinate-wise Trimmed Mean Defense before aggregation.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]): A list of tuples containing the number of samples
                and gradients for each client.
            extra_auxiliary_info (Any, optional): Additional auxiliary information. Default is None.

        Returns:
            OrderedDict: The aggregated global model after applying Coordinate-wise Trimmed Mean Defense.
        """
        if self.beta > 1 / 2 or self.beta < 0:
            raise ValueError("The bound of 'beta' is [0, 1/2)")
        return trimmed_mean(raw_client_grad_list, int(self.beta * len(raw_client_grad_list)))
