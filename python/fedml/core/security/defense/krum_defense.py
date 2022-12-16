from collections import OrderedDict

import torch
from typing import Callable, List, Tuple, Dict, Any
from .defense_base import BaseDefenseMethod
from ..common import utils

"""
defense @ server, added by Xiaoyang, Chulin, 07/09/2022
"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
https://arxiv.org/pdf/1703.02757.pdf
"Distributed momentum for byzantine-resilient stochastic gradient descent"
https://infoscience.epfl.ch/record/287261
"""


class KrumDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.config = config
        self.byzantine_client_num = config.byzantine_client_num

        # krum_param_m = 1: krum; krum_param_m > 1: multi-krum
        self.krum_param_m = 1  # krum
        if hasattr(config, "krum_param_m") and isinstance(config.krum_param_m, int):
            self.krum_param_m = config.krum_param_m

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        num_client = len(raw_client_grad_list)
        # in the Krum paper, it says 2 * byzantine_client_num + 2 < client #
        if not 2 * self.byzantine_client_num + 2 <= num_client - self.krum_param_m:
            raise ValueError(
                "byzantine_client_num conflicts with requirements in Krum: 2 * byzantine_client_num + 2 < client number - krum_param_m"
            )

        vec_local_w = [
            utils.vectorize_weight(raw_client_grad_list[i][1])
            for i in range(0, num_client)
        ]
        krum_scores = self._compute_krum_score(vec_local_w)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0 : self.krum_param_m]
        return [raw_client_grad_list[i] for i in score_index]

    def _compute_krum_score(self, vec_grad_list):
        krum_scores = []
        num_client = len(vec_grad_list)
        for i in range(0, num_client):
            dists = []
            for j in range(0, num_client):
                if i != j:
                    dists.append(
                        utils.compute_euclidean_distance(
                            vec_grad_list[i], vec_grad_list[j]
                        ).item() ** 2
                    )
            dists.sort()  # ascending
            score = dists[0 : num_client - self.byzantine_client_num - 2]
            krum_scores.append(sum(score))
        return krum_scores
