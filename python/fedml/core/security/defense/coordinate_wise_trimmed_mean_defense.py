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
    def __init__(self, config):
        self.beta = config.beta  # fraction of trimmed values; total trimmed values: client_num * beta * 2


    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        if self.beta > 1 / 2 or self.beta < 0:
            raise ValueError("the bound of beta is [0, 1/2)")
        return trimmed_mean(raw_client_grad_list, int(self.beta * len(raw_client_grad_list)))
