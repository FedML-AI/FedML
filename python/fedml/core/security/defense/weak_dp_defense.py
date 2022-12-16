from collections import OrderedDict
from typing import Callable, List, Tuple, Dict, Any
import torch
from .defense_base import BaseDefenseMethod

"""
defense @ server, added by Shanshan
"""


class WeakDPDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.config = config
        self.stddev = config.stddev  # for weak DP defenses

    def run(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ) -> Dict:
        new_grad_list = []
        for (sample_num, local_w) in raw_client_grad_list:
            new_w = self._add_noise(local_w)
            new_grad_list.append((sample_num, new_w))
        return base_aggregation_func(self.config, new_grad_list)  # avg_params

    def _add_noise(self, param):
        dp_param = dict()
        for k in param.keys():
            dp_param[k] = param[k] + torch.randn(param[k].size()) * self.stddev
        return dp_param
