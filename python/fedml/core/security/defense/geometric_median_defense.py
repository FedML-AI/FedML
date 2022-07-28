import math
from typing import Callable, List, Tuple, Dict, Any

import numpy as np

from ..common.bucket import Bucket
from ..common.utils import compute_middle_point, compute_euclidean_distance
from ...security.defense.defense_base import BaseDefenseMethod

"""
defense @ server with aggregation, added by Shanshan, 07/01/2022
"Distributed statistical machine learning in adversarial settings: Byzantine gradient descent. "
https://dl.acm.org/doi/pdf/10.1145/3154503

Steps: 
(1) divide m working machines into k batches,
(2) take the average of local gradients in each batch
(3) take the geometric median of those k batch means.
With the aggregated gradient, the parameter server performs a gradient descent update.
"""


class GeometricMedianDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.byzantine_client_num = config.byzantine_client_num
        self.client_num_per_round = config.client_num_per_round
        # 2(1 + ε )q ≤ batch_num ≤ client_num_per_round
        # trade-off between accuracy & robustness:
        #       larger batch_num --> more Byzantine robustness, larger estimation error.
        self.batch_num = config.batch_num
        if self.byzantine_client_num == 0:
            self.batch_num = 1
        self.batch_size = math.ceil(self.client_num_per_round / self.batch_num)

    def run(
        self,
        raw_client_grad_list: List[Tuple[float, Dict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ):
        batch_grad_list = Bucket.bucketization(raw_client_grad_list, self.batch_size)
        (num0, avg_params) = batch_grad_list[0]
        alphas = {alpha for (alpha, params) in batch_grad_list}
        alphas = {alpha / sum(alphas, 0.0) for alpha in alphas}
        for k in avg_params.keys():
            batch_grads = [params[k] for (alpha, params) in batch_grad_list]
            avg_params[k] = self._compute_geometric_median(alphas, batch_grads)
        return avg_params

    @staticmethod
    def _compute_geometric_median(alphas, batch_grads):
        """
        Implementation of Weiszfeld's algorithm.
        Reference:  (1) https://github.com/krishnap25/RFA/blob/master/models/model.py
                    (2) https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/geomed.py
        our contribution: (07/01/2022)
        1) fix one bug in (1): (1) can not correctly compute a weighted average. The function weighted_average_oracle
        returns zero.
        2) fix one bug in (2): (2) can not correctly handle multidimensional tensors.
        3) reconstruct the code.
        """
        eps = 1e-5
        ftol = 1e-10
        middle_point = compute_middle_point(alphas, batch_grads)
        val = sum(
            [
                alpha * compute_euclidean_distance(middle_point, p)
                for alpha, p in zip(alphas, batch_grads)
            ]
        )
        for i in range(100):
            prev_median, prev_obj_val = middle_point, val
            alphas = np.asarray(
                [
                    max(
                        eps,
                        alpha
                        / max(eps, compute_euclidean_distance(middle_point, a_batch_w)),
                    )
                    for alpha, a_batch_w in zip(alphas, batch_grads)
                ]
            )
            alphas = alphas / alphas.sum()
            middle_point = compute_middle_point(alphas, batch_grads)
            val = sum(
                [
                    alpha * compute_euclidean_distance(middle_point, p)
                    for alpha, p in zip(alphas, batch_grads)
                ]
            )
            if abs(prev_obj_val - val) < ftol * val:
                break
        return middle_point

    @staticmethod
    def compute_obj(alphas, batch_w, middle_point):
        return sum(
            [
                alpha * compute_euclidean_distance(middle_point, p)
                for alpha, p in zip(alphas, batch_w)
            ]
        )
