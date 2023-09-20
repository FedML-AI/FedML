from collections import OrderedDict
from typing import Callable, List, Tuple, Dict, Any
from ..common.utils import compute_geometric_median
from ...security.defense.defense_base import BaseDefenseMethod

"""
added by Shanshan
"RFA: Robust Aggregation for Federated Learning. "
https://arxiv.org/pdf/1912.13445.pdf
Compute a geometric median in aggreagtion
"""


class RFADefense(BaseDefenseMethod):
    """
    Robust Aggregation for Federated Learning (RFA) Defense.

    This defense method computes a geometric median in aggregation.

    Args:
        config: Configuration parameters (currently unused).

    Attributes:
        None

    Methods:
        defend_on_aggregation(
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            base_aggregation_func: Callable = None,
            extra_auxiliary_info: Any = None,
        ) -> OrderedDict:
        Defend against potential adversarial behavior during aggregation.

    References:
        - "RFA: Robust Aggregation for Federated Learning."
          https://arxiv.org/pdf/1912.13445.pdf
    """

    def __init__(self, config):
        """
        Initialize the RFADefense.

        Args:
            config: Configuration parameters (currently unused).
        """
        pass

    def defend_on_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ) -> OrderedDict:
        """
        Defend against potential adversarial behavior during aggregation.

        This method computes a geometric median aggregation of client gradients.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]):
                List of tuples containing client gradients as OrderedDict.
            base_aggregation_func (Callable, optional):
                Base aggregation function (currently unused).
            extra_auxiliary_info (Any, optional):
                Extra auxiliary information (currently unused).

        Returns:
            OrderedDict:
                Aggregated parameters after applying the defense.

        Notes:
            This defense method computes a geometric median aggregation of client gradients.
        """
        (num0, avg_params) = raw_client_grad_list[0]
        weights = {num for (num, params) in raw_client_grad_list}
        weights = {weight / sum(weights, 0.0) for weight in weights}
        for k in avg_params.keys():
            client_grads = [params[k] for (_, params) in raw_client_grad_list]
            avg_params[k] = compute_geometric_median(weights, client_grads)
        return avg_params
