import math
import torch
import numpy as np
from .defense_base import BaseDefenseMethod

from ..common.utils import vectorize_weight, is_weight_param


"""
defense @ client, added by Chulin
Distributed momentum for byzantine-resilient stochastic gradient descent
"""


class MultiKrumDefense(BaseDefenseMethod):
    def __init__(self, byzantine_client_num, client_num_per_round):
        self.byzantine_client_num = byzantine_client_num
        self.client_num_per_round = client_num_per_round

        assert client_num_per_round >= 2 * byzantine_client_num + 1, (
            "users_count>=2*corrupted_count + 3",
            client_num_per_round,
            byzantine_client_num,
        )

    def defend(self, local_w, global_w, refs=None):
        num_clients = len(local_w)
        (num0, localw0) = local_w[0]
        _len_local_params = vectorize_weight(localw0).shape[
            0
        ]  # lens of the flatted weights

        _params = np.zeros((num_clients, _len_local_params))
        for i in range(num_clients):
            _params[i] = vectorize_weight(local_w[i][1]).cpu().detach().numpy()

        alpha, multikrum_avg, distance_score = self._multi_krum(
            _params, self.client_num_per_round, self.byzantine_client_num
        )

        recons_local_w = {}
        index_bias = 0
        for item_index, (k, v) in enumerate(localw0.items()):
            if is_weight_param(k):
                recons_local_w[k] = torch.from_numpy(
                    multikrum_avg[index_bias : index_bias + v.numel()]
                ).view(
                    v.size()
                )  # todo: gpu/cpu issue for torch
                index_bias += v.numel()
            else:
                recons_local_w[k] = v

        return recons_local_w

    # Returns the index of the row that should be used in Krum
    def _multi_krum(self, deltas, n, clip):
        # assume deltas is an array of size group * d
        distance_scores = self._get_krum_scores(deltas, n - clip)
        good_idx = np.argpartition(distance_scores, n - clip)[: (n - clip)]
        alpha = np.zeros(len(deltas))
        alpha[good_idx] = 1

        return alpha, np.mean(deltas[good_idx], axis=0), distance_scores

    @staticmethod
    def _get_krum_scores(X, groupsize):
        krum_scores = np.zeros(len(X))
        # Calculate distances
        distances = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(X**2, axis=1)[None]
            - 2 * np.dot(X, X.T)
        )

        for i in range(len(X)):
            krum_scores[i] = np.sum(np.sort(distances[i])[1 : (groupsize - 1)])

        return krum_scores
