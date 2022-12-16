import math
from collections import OrderedDict
from typing import Callable, List, Tuple, Dict, Any
from ..common.utils import trimmed_mean
from ..defense.defense_base import BaseDefenseMethod

"""
defense with aggregation, added by Shanshan, 07/13/2022
SLSGD: Secure and efficient distributed on-device machine learning
https://arxiv.org/pdf/1903.06996.pdf

This approach has two stage: 1. process model list, and 2. aggregation

In stage 1, users can choose whether to trim the model list (option 2 when b > 0) or not (option 1 or option 2 when b = 0).
In the first case, the steps are as follows: 
    1) compute a score for gradients from clients; 
    2) sort the model list with the scores
    3) (trim) remove the first b gradients and the last b gradients from the model list.
In the second case, the algorithm does nothing to the model list.

In stage 2, a user can set alpha (i.e., the weight of moving average) between 0 and 1 
and does an aggregation with the averaged model and the global model in the last iteration using aggregate(). 
Specifically, alpha = 1 indicates the new global model is set to the new average model, 
and alpha = 0 indicates the global model is identical to the old one.

"""


class SLSGDDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.b = config.trim_param_b  # parameter of trimmed mean
        if config.alpha > 1 or config.alpha < 0:
            raise ValueError("the bound of alpha is [0, 1]")
        self.alpha = config.alpha
        self.option_type = config.option_type
        self.config = config

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        if self.b > math.ceil(len(raw_client_grad_list) / 2) - 1 or self.b < 0:
            raise ValueError(
                "the bound of b is [0, {}])".format(
                    math.ceil(len(raw_client_grad_list) / 2) - 1
                )
            )
        if self.option_type != 1 and self.option_type != 2:
            raise Exception("Such option type does not exist!")
        if self.option_type == 2:
            raw_client_grad_list = trimmed_mean(
                raw_client_grad_list, self.b
            )  # process model list
        return raw_client_grad_list

    def defend_on_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ):
        global_model = extra_auxiliary_info
        avg_params = base_aggregation_func(args=self.config, raw_grad_list=raw_client_grad_list)
        for k in avg_params.keys():
            avg_params[k] = (1 - self.alpha) * global_model[
                k
            ] + self.alpha * avg_params[k]
        return avg_params
