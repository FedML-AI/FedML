from collections import OrderedDict
from typing import Callable, List, Tuple, Dict, Any
import torch
from .defense_base import BaseDefenseMethod

"""
defense @ server, added by Shanshan
"""


class WeakDPDefense(BaseDefenseMethod):
    """
    Weak Differential Privacy (DP) Defense for Federated Learning.

    This defense method adds weak differential privacy noise to client gradients to enhance privacy.

    Args:
        config: Configuration object containing defense parameters.

    Methods:
        run(
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            base_aggregation_func: Callable = None,
            extra_auxiliary_info: Any = None,
        ) -> Dict:
        Run the weak DP defense.

    Attributes:
        config: Configuration object containing defense parameters.
        stddev: Standard deviation for adding noise to gradients.
    """

    def __init__(self, config):
        self.config = config
        self.stddev = config.stddev  # for weak DP defenses

    def run(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ) -> Dict:
        """
        Run the weak DP defense by adding noise to client gradients.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]):
                List of tuples containing client gradients as OrderedDict.
            base_aggregation_func (Callable, optional):
                Base aggregation function (currently unused).
            extra_auxiliary_info (Any, optional):
                Extra auxiliary information (currently unused).

        Returns:
            Dict: Dictionary containing aggregated model parameters with added noise.
        """
        new_grad_list = []
        for (sample_num, local_w) in raw_client_grad_list:
            new_w = self._add_noise(local_w)
            new_grad_list.append((sample_num, new_w))
        return base_aggregation_func(self.config, new_grad_list)  # avg_params

    def _add_noise(self, param):
        """
        Add Gaussian noise to the parameters.

        Args:
            param (OrderedDict): Client parameters.

        Returns:
            OrderedDict: Parameters with added noise.
        """
        dp_param = dict()
        for k in param.keys():
            dp_param[k] = param[k] + torch.randn(param[k].size()) * self.stddev
        return dp_param
