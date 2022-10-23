import math
import numpy as np
from .defense_base import BaseDefenseMethod
from typing import Callable, List, Tuple, Dict, Any
from ..common.utils import compute_euclidean_distance, compute_middle_point, compute_krum_score
import torch


def compute_gaussian_distribution(score_list):
    n = len(score_list)
    mu = sum(list(score_list)) / n
    temp = 0

    for i in range(len(score_list)):
        temp = (((score_list[i] - mu) ** 2) / (n - 1)) + temp
    sigma = math.sqrt(temp)
    return mu, sigma


class ThreeSigmaKrumDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.memory = None
        self.iteration_num = 0
        self.median = None

        if hasattr(config, "pretraining_round_num") and isinstance(
            config.pretraining_round_num, int
        ):
            self.pretraining_round_number = config.pretraining_round_num
        else:
            self.pretraining_round_number = 2
        # ----------------- params for normal distribution ----------------- #
        self.upper_bound = 0
        self.bound_param = 1  # values outside mu +- sigma are outliers

    def run(
        self,
        raw_client_grad_list: List[Tuple[float, Dict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ):
        grad_list = self.defend_before_aggregation(
            raw_client_grad_list, extra_auxiliary_info
        )
        return self.defend_on_aggregation(
            grad_list, base_aggregation_func, extra_auxiliary_info
        )

    ###################### version 3: re-compute gaussian distribution each round
    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, Dict]],
        extra_auxiliary_info: Any = None,
    ):
        client_scores = self.compute_client_scores(raw_client_grad_list)
        print(f"client scores = {client_scores}")
        mu, sigma = compute_gaussian_distribution(client_scores)
        self.upper_bound = mu + self.bound_param * sigma
        print(f"mu = {mu}, sigma = {sigma}, upperbound = {self.upper_bound}")
        new_client_models, _ = self.kick_out_poisoned_local_models(client_scores, raw_client_grad_list)
        importance_feature_list = self._get_importance_feature(new_client_models)
        alphas = [1 / len(importance_feature_list)] * len(importance_feature_list)
        self.median = compute_middle_point(alphas, importance_feature_list)
        self.iteration_num += 1
        return new_client_models

    ##################### version 2: remove poisoned model scores in score list
    # def defend_before_aggregation(
    #     self,
    #     raw_client_grad_list: List[Tuple[float, Dict]],
    #     extra_auxiliary_info: Any = None,
    # ):
    #     client_scores = self.compute_client_scores(raw_client_grad_list)
    #     print(f"client scores = {client_scores}")
    #     if self.iteration_num < self.pretraining_round_number:
    #         mu, sigma = compute_gaussian_distribution(self.score_list, client_scores)
    #         self.upper_bound = mu + self.bound_param * sigma
    #         print(f"mu = {mu}, sigma = {sigma}, upperbound = {self.upper_bound}")
    #         new_client_models, client_scores = self.kick_out_poisoned_local_models(client_scores, raw_client_grad_list)
    #         print(f"new scores after kicking out = {client_scores}")
    #         self.score_list.extend(list(client_scores))
    #         mu, sigma = compute_gaussian_distribution(self.score_list, [])
    #         self.upper_bound = mu + self.bound_param * sigma
    #         print(f"mu = {mu}, sigma = {sigma}, upperbound = {self.upper_bound}")
    #     else:
    #         new_client_models, _ = self.kick_out_poisoned_local_models(client_scores, raw_client_grad_list)
    #     self.iteration_num += 1
    #     return new_client_models

    ###################### version 1: do not remove poisoned model scores in score list
    # def defend_before_aggregation(
    #     self,
    #     raw_client_grad_list: List[Tuple[float, Dict]],
    #     extra_auxiliary_info: Any = None,
    # ):
    #     client_scores = self.compute_client_scores(raw_client_grad_list)
    #     print(f"client scores = {client_scores}")
    #
    #     if self.iteration_num < self.pretraining_round_number:
    #         self.score_list.extend(list(client_scores))
    #         self.mu, self.sigma = self.compute_gaussian_distribution_old()
    #         self.upper_bound = self.mu + self.bound_param * self.sigma
    #         self.iteration_num += 1
    #
    #     for i in range(len(client_scores) - 1, -1, -1):
    #         if client_scores[i] > self.upper_bound:
    #      # we do not remove the score in self.score_list to avoid mis-deleting due to severe non-iid among clients
    #             raw_client_grad_list.pop(i)
    #             print(f"pop -- i = {i}")
    #     return raw_client_grad_list

    def kick_out_poisoned_local_models(self, client_scores, raw_client_grad_list):
        print(f"upper bound = {self.upper_bound}")
        for i in range(
                len(client_scores) - 1, -1, -1
        ):  # traverse the score list in a reversed order
            if client_scores[i] > self.upper_bound:
                raw_client_grad_list.pop(i)
                client_scores.pop(i)
                print(f"pop -- i = {i}")
        return raw_client_grad_list, client_scores

    def compute_client_scores(self, raw_client_grad_list):
        if self.median is None:
            self.median = self.compute_median_with_krum(raw_client_grad_list)
        return self.l2_scores(raw_client_grad_list)

    def compute_median_with_krum(self, raw_client_grad_list):
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        krum_scores = compute_krum_score(importance_feature_list, client_num_after_trim=math.floor(len(raw_client_grad_list) / 2))
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: math.floor(len(raw_client_grad_list) / 2)]
        honest_importance_feature_list = [
            importance_feature_list[i] for i in score_index
        ]
        alphas = [1 / len(honest_importance_feature_list)] * len(honest_importance_feature_list)
        median = compute_middle_point(
            alphas, honest_importance_feature_list
        )
        return median

    def l2_scores(self, raw_client_grad_list):
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        scores = []
        for feature in importance_feature_list:
            score = compute_euclidean_distance(torch.Tensor(feature), self.median)
            scores.append(score)
        return scores

    def _get_importance_feature(self, raw_client_grad_list):
        # print(f"raw_client_grad_list = {raw_client_grad_list}")
        # Foolsgold uses the last layer's gradient/weights as the importance feature.
        ret_feature_vector_list = []
        for idx in range(len(raw_client_grad_list)):
            raw_grad = raw_client_grad_list[idx]
            (p, grads) = raw_grad

            # Get last key-value tuple
            (weight_name, importance_feature) = list(grads.items())[-2]
            # print(importance_feature)
            feature_len = np.array(
                importance_feature.cpu().data.detach().numpy().shape
            ).prod()
            feature_vector = np.reshape(
                importance_feature.cpu().data.detach().numpy(), feature_len
            )
            ret_feature_vector_list.append(feature_vector)
        return ret_feature_vector_list
