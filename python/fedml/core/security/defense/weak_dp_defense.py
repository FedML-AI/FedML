from typing import Callable, List, Tuple, Dict, Any
from wandb.wandb_torch import torch
from .defense_base import BaseDefenseMethod

"""
defense @ server, added by Shanshan

Coordinate-wise Median from "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates".
"""


class WeakDPDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.stddev = config.stddev  # for weak DP defenses

    def run(
        self,
        raw_client_grad_list: List[Tuple[float, Dict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ) -> Dict:
        new_grad_list = []
        for (sample_num, local_w) in raw_client_grad_list:
            new_w = self.add_noise(local_w)
            new_grad_list.append((sample_num, new_w))
        return base_aggregation_func(new_grad_list)  # avg_params

    def add_noise(self, local_weight):
        gaussian_noise = torch.randn(local_weight.size()) * self.stddev
        dp_weight = local_weight + gaussian_noise
        return dp_weight
