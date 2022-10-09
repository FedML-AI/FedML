import math
import numpy as np
from .defense_base import BaseDefenseMethod
from typing import Callable, List, Tuple, Dict, Any
from ..common.bucket import Bucket
from ..common.utils import compute_euclidean_distance, compute_middle_point
import torch


class ThreeSigmaKrumDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.memory = None
        self.iteration_num = 1
        self.score_list = []
        self.median = None

        if hasattr(config, "bucketing_batch_size") and isinstance(
            config.bucketing_batch_size, int
        ):
            self.bucketing_batch_size = config.bucketing_batch_size
        else:
            self.bucketing_batch_size = 1
        if hasattr(config, "pretraining_round_num") and isinstance(
            config.pretraining_round_num, int
        ):
            self.pretraining_round_number = config.pretraining_round_num
        else:
            self.pretraining_round_number = 2
        # ----------------- params for normal distribution ----------------- #
        self.mu = 0
        self.sigma = 0
        self.upper_bound = 0
        self.lower_bound = 0
        self.bound_param = 1  # values outside mu +- sigma are outliers

        if hasattr(config, "to_keep_higher_scores") and isinstance(
            config.to_keep_higher_scores, bool
        ):
            self.to_keep_higher_scores = config.to_keep_higher_scores
        else:
            self.to_keep_higher_scores = (
                False  # true or false, depending on the score algo
            )
        self.score_function = "l2"

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

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, Dict]],
        extra_auxiliary_info: Any = None,
    ):
        client_scores = self.compute_client_scores(raw_client_grad_list)
        print(f"client scores = {client_scores}")
        if self.iteration_num < self.pretraining_round_number:
            self.score_list.extend(list(client_scores))
            self.mu, self.sigma = self.compute_gaussian_distribution()
            # if self.mu + self.bound_param * self.sigma >= 0:
            self.upper_bound = self.mu + self.bound_param * self.sigma
            # if self.mu - self.bound_param * self.sigma <= 0:
            self.lower_bound = self.mu - self.bound_param * self.sigma
            self.iteration_num += 1

        for i in range(
            len(client_scores) - 1, -1, -1
        ):  # traverse the score list in a reversed order
            if (
                not self.to_keep_higher_scores and client_scores[i] > self.upper_bound
            ) or (self.to_keep_higher_scores and client_scores[i] < self.lower_bound):
                # here we do not remove the score in self.score_list to avoid mis-deleting
                # due to severe non-iid among clients
                raw_client_grad_list.pop(i)
                print(f"pop -- i = {i}")
        batch_grad_list = Bucket.bucketization(
            raw_client_grad_list, self.bucketing_batch_size
        )
        return batch_grad_list

    def compute_gaussian_distribution(self):
        n = len(self.score_list)
        mu = sum(list(self.score_list)) / n
        temp = 0

        for i in range(len(self.score_list)):
            temp = (((self.score_list[i] - mu) ** 2) / (n - 1)) + temp
        sigma = math.sqrt(temp)
        print(f"mu = {mu}, sigma = {sigma}")
        return mu, sigma

    def compute_client_scores(self, raw_client_grad_list):
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        if self.score_function == "l2":
            if self.median is None:
                # (num0, avg_params) = raw_client_grad_list[0]
                # alphas = {alpha for (alpha, params) in raw_client_grad_list}
                # alphas = {alpha / sum(alphas, 0.0) for alpha in alphas}
                krum_scores = self._compute_krum_score(importance_feature_list)

                score_index = torch.argsort(
                    torch.Tensor(krum_scores)
                ).tolist()  # indices; ascending
                score_index = score_index[0 : math.floor(len(raw_client_grad_list) / 2)]
                alphas = [1 / len(raw_client_grad_list)] * len(raw_client_grad_list)
                honest_importance_feature_list = [
                    importance_feature_list[i] for i in score_index
                ]
                self.median = compute_middle_point(
                    alphas, honest_importance_feature_list
                )
            return self.l2_scores(importance_feature_list)

    def l2_scores(self, importance_feature_list):
        scores = []
        for feature in importance_feature_list:
            score = compute_euclidean_distance(torch.Tensor(feature), self.median)
            scores.append(score)
        return scores

    def _compute_krum_score(self, vec_grad_list):
        krum_scores = []
        num_client = len(vec_grad_list)
        for i in range(0, num_client):
            dists = []
            for j in range(0, num_client):
                if i != j:
                    dists.append(
                        compute_euclidean_distance(
                            torch.Tensor(vec_grad_list[i]),
                            torch.Tensor(vec_grad_list[j]),
                        ).item() ** 2
                    )
            dists.sort()  # ascending
            score = dists[0 : math.floor(num_client / 2)]
            krum_scores.append(sum(score))
        return krum_scores

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
