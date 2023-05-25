from collections import OrderedDict
import torch
from typing import Callable, List, Tuple, Any
from .defense_base import BaseDefenseMethod
from ..common.utils import vectorize_weight

"""
Coordinate-wise Median from "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates".
This can be called at aggregate() of an Aggregator inplace of parameter averaging after \
model_list has been created
 """


class CoordinateWiseMedianDefense(BaseDefenseMethod):
    def __init__(self, config):
        pass

    def defend_on_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            base_aggregation_func: Callable = None,
            extra_auxiliary_info: Any = None,
    ):
        vectorized_params = []

        for i in range(0, len(raw_client_grad_list)):
            local_sample_number, local_model_params = raw_client_grad_list[i]
            vectorized_weight = vectorize_weight(local_model_params)
            vectorized_params.append(vectorized_weight.unsqueeze(-1))

        # concatenate all weights by the last dimension (number of clients)
        vectorized_params = torch.cat(vectorized_params, dim=-1)
        vec_median_params = torch.median(vectorized_params, dim=-1).values

        index = 0
        (num0, averaged_params) = raw_client_grad_list[0]
        for k, params in averaged_params.items():
            median_params = vec_median_params[index : index + params.numel()].view(
                params.size()
            )
            index += params.numel()
            averaged_params[k] = median_params

        return averaged_params

