import math

import numpy as np
import torch

from ...security.defense.defense_base import BaseDefenseMethod

"""
added by Shanshan, 07/01/2022
"Distributed statistical machine learning in adversarial settings: Byzantine gradient descent. "
https://dl.acm.org/doi/pdf/10.1145/3154503
"""

# (1) divide m working machines into k batches,
# (2) take the average of local gradients in each batch
# (3) take the geometric median of those k batch means.
# With the aggregated gradient, the parameter server performs a gradient descent update.




def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def _compute_geometric_median_objective(middle_point, points, alphas):
    return sum([alpha * _compute_euclidean_distance(middle_point, p) for alpha, p in zip(alphas, points)])


def _compute_middle_point(alphas, batch_w):
    return torch.sum(torch.vstack([a * a_batch_w for a, a_batch_w in zip(alphas, batch_w)]), dim=0)


class GeometricMedian(BaseDefenseMethod):
    def __init__(self, byzantine_client_num, client_num_per_round, batch_num):
        self.byzantine_client_num = byzantine_client_num
        self.client_num_per_round = client_num_per_round
        # 2(1 + ε )q ≤ batch_num ≤ client_num_per_round
        # trade-off between accuracy & robustness:
        #       larger batch_num --> more Byzantine robustness, larger estimation error.
        if self.byzantine_client_num == 0:
            self.batch_num = 1
        else:
            self.batch_num = batch_num

        self.batch_size = math.ceil(self.client_num_per_round/self.batch_num)

    def defend(self, local_w, global_w=None, refs = None):
        batch_w = []
        (num0, averaged_params) = local_w[0]
        print(num0, averaged_params)

        for k in averaged_params.keys():
            alphas = dict()  # weights for each avg local_w for each batch
            for batch_idx in range(0, math.ceil(len(local_w)/self.batch_size)):
                client_num = self._get_client_num_current_batch(batch_idx, local_w)
                training_num_for_batch = self._get_total_training_num_for_current_batch(batch_idx, client_num, local_w)
                alphas[batch_idx] = training_num_for_batch
                batch_weight = 0
                for i in range(0, client_num):
                    local_sample_number, local_model_params = local_w[batch_idx * self.batch_size + i]
                    w = local_sample_number / training_num_for_batch
                    if i == 0:
                        batch_weight = local_model_params[k] * w
                    else:
                        batch_weight += local_model_params[k] * w
                batch_w.append(batch_weight)
            print(f"batch_w={batch_w}")
            alphas = {batch_idx: alpha/sum(alphas.values(), 0.0) for batch_idx, alpha in alphas.items()}
            averaged_params[k] = self.compute_geometric_median(alphas, batch_w)
        return averaged_params


    def compute_geometric_median(self, alphas, batch_w):
        """
        Implemention of Weiszfeld's algorithm.
        Reference:  https://github.com/krishnap25/RFA/blob/master/models/model.py
                    https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/geomed.py
        """
        eps = 1e-5
        ftol = 1e-10
        middle_point = _compute_middle_point(alphas, batch_w)
        val = _compute_geometric_median_objective(middle_point, batch_w, alphas)
        for i in range(100):
            prev_median, prev_obj_val = middle_point, val
            alphas = np.asarray(
                [max(eps, alpha / max(eps, _compute_euclidean_distance(middle_point, a_batch_w))) for alpha, a_batch_w in
                 zip(alphas, batch_w)])
            alphas = alphas / alphas.sum()
            middle_point = _compute_middle_point(alphas, batch_w)
            val = _compute_geometric_median_objective(middle_point, batch_w, alphas)

            if abs(prev_obj_val - val) < ftol * val:
                break
        return val

    def _get_total_training_num_for_current_batch(self, batch_idx, current_batch_size, local_w):
        training_num_for_batch = 0
        for i in range(0, current_batch_size):
            local_sample_number, local_model_params = local_w[batch_idx * self.batch_size + i]
            training_num_for_batch += local_sample_number
        return training_num_for_batch

    def _get_client_num_current_batch(self, batch_idx, local_w):
        current_batch_size = self.batch_size
        if batch_idx == math.ceil(len(local_w) / self.batch_size) - 1:
            current_batch_size = len(local_w) - (batch_idx * self.batch_size)
        return current_batch_size
