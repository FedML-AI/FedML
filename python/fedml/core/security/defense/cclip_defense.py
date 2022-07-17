import random
from typing import Callable, List, Tuple, Dict, Any

from .defense_base import BaseDefenseMethod
from ..common import utils

"""
defense @ server, added by Xiaoyang, 07/10/2022
"Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing"
https://arxiv.org/pdf/2006.09365.pdf
"""


class CClipDefense(BaseDefenseMethod):

    def __init__(self, tau, bucket_size=1):
        self.tau = tau  # clipping raduis
        # element # in each bucket; a grad_list is partitioned into floor(len(grad_list)/bucket_size) buckets
        self.bucket_size = bucket_size

    def run(self, base_aggregation_func: Callable, raw_client_grad_list: List[Tuple[int, Dict]],
            extra_auxiliary_info: Any = None) -> Dict:
        avg_params = base_aggregation_func(raw_client_grad_list)
        initial_guess = self._compute_an_initial_guess(raw_client_grad_list)
        for k in avg_params.keys():
            avg_params[k] = initial_guess[k] + avg_params[k]
        return avg_params

    @staticmethod
    def _compute_an_initial_guess(client_grad_list):
        # randomly select a gradient as the initial guess
        return client_grad_list[random.randint(0, len(client_grad_list))][1]

    def _compute_cclip_score(self, local_w, refs):
        cclip_score = []
        num_client = len(local_w)
        for i in range(0, num_client):
            dist = utils.compute_euclidean_distance(local_w[i][1], refs).item() + 1e-8
            score = min(1, self.tau / dist)
            cclip_score.append(score)
        return cclip_score
