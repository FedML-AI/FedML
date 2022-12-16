from collections import OrderedDict
from typing import Callable, List, Tuple, Any
import numpy as np
from .defense_base import BaseDefenseMethod
from ..common import utils
from ..common.bucket import Bucket

"""
defense @ server, added by Xiaoyang, 07/10/2022
"Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing"
https://arxiv.org/pdf/2006.09365.pdf
"""


class CClipDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.config = config
        self.tau = config.tau  # clipping raduis
        # element # in each bucket; a grad_list is partitioned into floor(len(grad_list)/bucket_size) buckets
        self.bucket_size = config.bucket_size

    def run(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ) -> OrderedDict:
        client_grad_buckets = Bucket.bucketization(
            raw_client_grad_list, self.bucket_size
        )
        initial_guess = self._compute_an_initial_guess(client_grad_buckets)
        bucket_num = len(client_grad_buckets)
        vec_local_w = [
            (
                client_grad_buckets[i][0],
                utils.vectorize_weight(client_grad_buckets[i][1]),
            )
            for i in range(bucket_num)
        ]
        vec_refs = utils.vectorize_weight(initial_guess)
        cclip_score = self._compute_cclip_score(vec_local_w, vec_refs)
        new_grad_list = []
        for i in range(bucket_num):
            tuple = dict()
            sample_num, bucket_params = client_grad_buckets[i]
            for k in bucket_params.keys():
                tuple[k] = (bucket_params[k] - initial_guess[k]) * cclip_score[i]
            new_grad_list.append((sample_num, tuple))
        print(f"cclip_score = {cclip_score}")

        avg_params = base_aggregation_func(self.config, new_grad_list)
        for k in avg_params.keys():
            avg_params[k] = initial_guess[k] + avg_params[k]
        return avg_params

    @staticmethod
    def _compute_an_initial_guess(client_grad_list):
        # randomly select a gradient as the initial guess
        return client_grad_list[np.random.randint(0, len(client_grad_list))][1]

    def _compute_cclip_score(self, local_w, refs):
        cclip_score = []
        num_client = len(local_w)
        for i in range(0, num_client):
            dist = utils.compute_euclidean_distance(local_w[i][1], refs).item() + 1e-8
            score = min(1, self.tau / dist)
            cclip_score.append(score)
        return cclip_score
