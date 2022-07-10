import torch

from .defense_base import BaseDefenseMethod
from ..common import utils

"""
defense @ server, added by Xiaoyang, 07/09/2022
"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
https://arxiv.org/pdf/1703.02757.pdf
"""


class KrumDefense(BaseDefenseMethod):
    def __init__(self, k, multi=False):
        self.k = k          # assume there are k byzantine clients
        self.multi = multi  # krum or multi-krum

    def defend(self, local_w, global_w=None, refs=None):
        num_client = len(local_w)
        vec_local_w = [(local_w[i][0], utils.vectorize_weight(local_w[i][1])) for i in range(0, num_client)]
        print(vec_local_w)
        krum_score = self._compute_krum_score(vec_local_w)
        index = torch.argsort(torch.Tensor(krum_score)).tolist()
        if self.multi is True:
            index = index[0: num_client - self.k]
        else:
            index = index[0: 1]

        local_w = [local_w[i] for i in index]

        return local_w, krum_score

    def _compute_krum_score(self, local_w):
        krum_score = []
        num_client = len(local_w)
        for i in range(0, num_client):
            dists = []
            for j in range(0, num_client):
                if i == j:
                    continue
                dists.append(utils.compute_euclidean_distance(local_w[i][1], local_w[j][1]).item())
            dists.sort()
            score = dists[0: num_client - self.k]
            krum_score.append(sum(score))

        return krum_score
