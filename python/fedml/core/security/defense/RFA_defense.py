from collections import OrderedDict
from typing import Callable, List, Tuple, Dict, Any
from ..common.utils import compute_geometric_median
from ...security.defense.defense_base import BaseDefenseMethod
import fedml

"""
added by Shanshan
"RFA: Robust Aggregation for Federated Learning. "
https://arxiv.org/pdf/1912.13445.pdf
Compute a geometric median in aggreagtion
"""


class RFADefense(BaseDefenseMethod):
    def __init__(self, config):
        self.device = fedml.device.get_device(config)

    def defend_on_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            base_aggregation_func: Callable = None,
            extra_auxiliary_info: Any = None,
    ):
        (num0, avg_params) = raw_client_grad_list[0]
        weights = {num for (num, params) in raw_client_grad_list}
        weights = {weight / sum(weights, 0.0) for weight in weights}
        for k in avg_params.keys():
            client_grads = [params[k] for (_, params) in raw_client_grad_list]
            avg_params[k] = compute_geometric_median(weights, client_grads)
        return avg_params
