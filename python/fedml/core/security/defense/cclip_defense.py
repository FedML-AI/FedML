import torch

from .defense_base import BaseDefenseMethod
from ..common import utils

"""
defense @ server, added by Xiaoyang, 07/10/2022
"Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing"
https://arxiv.org/pdf/2006.09365.pdf
"""


class CClipDefense(BaseDefenseMethod):
    def __init__(self, tau):
        self.tau = tau  # clipping raduis

    def defend(self, client_grad_list, refs):
        num_client = len(client_grad_list)
        vec_local_w = [
            (client_grad_list[i][0], utils.vectorize_weight(client_grad_list[i][1]))
            for i in range(0, num_client)
        ]
        print(vec_local_w)
        vec_refs = utils.vectorize_weight(refs)
        cclip_score = self._compute_cclip_score(vec_local_w, vec_refs)

        _, averaged_params = client_grad_list[0]
        for k in averaged_params.keys():
            averaged_params[k] = refs[k]
            for i in range(0, num_client):
                _, local_model_params = client_grad_list[i]
                averaged_params[k] += (local_model_params[k] - refs[k]) * cclip_score[i]

        return averaged_params, cclip_score

    def _compute_cclip_score(self, local_w, refs):
        cclip_score = []
        num_client = len(local_w)
        for i in range(0, num_client):
            dist = utils.compute_euclidean_distance(local_w[i][1], refs).item() + 1e-8
            score = min(1, self.tau / dist)
            cclip_score.append(score)

        return cclip_score
