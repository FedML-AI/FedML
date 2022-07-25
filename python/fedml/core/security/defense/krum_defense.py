import torch
from typing import Callable, List, Tuple, Dict, Any
from .defense_base import BaseDefenseMethod
from ..common import utils

"""
defense @ server, added by Xiaoyang, 07/09/2022
"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
https://arxiv.org/pdf/1703.02757.pdf
"""


class KrumDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.k = config.byzantine_client_num  # assume there are k byzantine clients
        if config.multi is None:
            self.multi = False
        else:
            self.multi = config.multi  # krum or multi-krum

    def run(
        self,
        raw_client_grad_list: List[Tuple[float, Dict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ) -> Dict:
        num_client = len(raw_client_grad_list)
        vec_local_w = [
            (raw_client_grad_list[i][0], utils.vectorize_weight(raw_client_grad_list[i][1]))
            for i in range(0, num_client)
        ]
        print(vec_local_w)
        krum_score = self._compute_krum_score(vec_local_w)
        index = torch.argsort(torch.Tensor(krum_score)).tolist()
        if self.multi is True:
            index = index[0:num_client - self.k]
        else:
            index = index[0:1]

        grad_list = [raw_client_grad_list[i] for i in index]
        print(f"krum_scores = {krum_score}")
        return base_aggregation_func(grad_list)  # avg_params

    def _compute_krum_score(self, local_w):
        krum_score = []
        num_client = len(local_w)
        for i in range(0, num_client):
            dists = []
            for j in range(0, num_client):
                if i == j:
                    continue
                dists.append(
                    utils.compute_euclidean_distance(
                        local_w[i][1], local_w[j][1]
                    ).item()
                )
            dists.sort()
            score = dists[0 : num_client - self.k]
            krum_score.append(sum(score))

        return krum_score
