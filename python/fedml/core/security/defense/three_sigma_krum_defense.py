import math
import numpy as np
from scipy import spatial
from .defense_base import BaseDefenseMethod
from typing import Callable, List, Tuple, Dict, Any
from .foolsgold_defense import FoolsGoldDefense
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
        self.average = None
        self.score_list = []
        self.flag_mu = 1000
        self.flag_sigma = 400

        # if hasattr(config, "pretraining_round_num") and isinstance(
        #         config.pretraining_round_num, int
        # ):
        #     self.pretraining_round_number = config.pretraining_round_num
        # else:
        #     self.pretraining_round_number = 2
        # ----------------- params for normal distribution ----------------- #
        self.upper_bound = 0
        # self.bound_param = 1  # values outside mu +- sigma are outliers
        if hasattr(config, "bound_param") and isinstance(
            config.bound_param, float
        ):
            self.bound_param = config.bound_param
        else:
            self.bound_param = 1
        if hasattr(config, "score_function") and config.score_function in ["l2", "krum", "cosine", "krum_cosine", "l2_cosine", "l1", "l1_cosine"]:
            self.score_function = config.score_function
            print(f"score function = {config.score_function}")
        else:
            raise ValueError("score function error")


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
        new_client_models = []
        if self.score_function in ["l2", "l2_cosine"]:
            if self.average is None:
                self.average = self.compute_median_with_krum(raw_client_grad_list)

            client_scores = self.compute_l2_scores(raw_client_grad_list)

            if self.score_function == "l2_cosine":
                mu, sigma = compute_gaussian_distribution(client_scores)
                print(f"client score mu = {mu}, sigma = {sigma}")
                cosine_scores = self.compute_client_cosine_scores_with_avg(raw_client_grad_list)
                mu_cos, sigma_cos = compute_gaussian_distribution(cosine_scores)

                a = math.sqrt((self.flag_sigma ** 2) / (sigma ** 2))
                b = self.flag_mu - a * mu
                client_scores = [a * client_score + b for client_score in client_scores]
                a = math.sqrt((self.flag_sigma ** 2) / (sigma_cos ** 2))
                b = self.flag_mu - a * mu_cos
                cosine_scores = [a * cosine_score + b for cosine_score in cosine_scores]

                print(f"old cosine distribution = {mu_cos}, {sigma_cos}")
                mu_cos, sigma_cos = compute_gaussian_distribution(cosine_scores)
                print(f"new cosine distribution = {mu_cos}, {sigma_cos}")
                print(f"l2 socres = {client_scores}")
                print(f"cosine_scores = {cosine_scores}")
                client_scores = [0.5 * client_scores[i] + 0.5 * (cosine_scores[i]) for i in range(len(client_scores))]
            print(f"client scores = {client_scores}")

            mu, sigma = compute_gaussian_distribution(client_scores)
            self.upper_bound = mu + self.bound_param * sigma  # Decimal(mu + self.bound_param * sigma).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")
            print(f"upperbound 1 = {self.upper_bound}")
            # self.upper_bound = float('%.2f' % self.upper_bound)
            # self.upper_bound = Decimal(self.upper_bound).quantize(Decimal("0.01"), rounding = "ROUND_UP") #"ROUND_HALF_UP")
            print(f"client socres = {client_scores}")
            print(f"mu = {mu}, sigma = {sigma}, upperbound = {self.upper_bound}")
            new_client_models, _ = self.kick_out_poisoned_local_models(client_scores, raw_client_grad_list)
            importance_feature_list = self._get_importance_feature(new_client_models)
            self.average = self.compute_an_average_point(importance_feature_list)
            # krum, cosine methods do not kick out malicious scores thus the results may be not good
            # optimization: use average to compute a cosine value
        return new_client_models

    def compute_an_average_point(self, importance_feature_list):
        alphas = [1 / len(importance_feature_list)] * len(importance_feature_list)
        return compute_middle_point(alphas, importance_feature_list)

    ##################### version 2: remove poisoned model scores in score list
    # def defend_before_aggregation(
    #     self,
    #     raw_client_grad_list: List[Tuple[float, Dict]],
    #     extra_auxiliary_info: Any = None,
    # ):
    #     if self.median is None:
    #         self.median = self.compute_median_with_krum(raw_client_grad_list)
    #     client_scores = self.compute_scores(raw_client_grad_list)
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
    #     if self.median is None:
    #         self.median = self.compute_median_with_krum(raw_client_grad_list)
    #     client_scores = self.compute_scores(raw_client_grad_list)
    #     print(f"client scores = {client_scores}")
    #
    #     if self.iteration_num < self.pretraining_round_number:
    #         self.score_list.extend(list(client_scores))
    #         self.mu, self.sigma = compute_gaussian_distribution(self.score_list)
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

    def compute_median_with_krum(self, raw_client_grad_list):
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        krum_scores = compute_krum_score(importance_feature_list,
                                         client_num_after_trim=math.floor(len(raw_client_grad_list) / 2))
        score_index = torch.argsort(torch.Tensor(krum_scores)).tolist()  # indices; ascending
        score_index = score_index[0: math.floor(len(raw_client_grad_list) / 2)]
        honest_importance_feature_list = [importance_feature_list[i] for i in score_index]
        return self.compute_an_average_point(honest_importance_feature_list)

    def compute_l2_scores(self, raw_client_grad_list):
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        scores = []
        for feature in importance_feature_list:
            score = compute_euclidean_distance(torch.Tensor(feature), self.average)
            scores.append(score)
        return scores

    def compute_l1_scores(self, raw_client_grad_list):
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        scores = []
        for feature in importance_feature_list:
            score = np.linalg.norm((torch.Tensor(feature) - self.average), ord=1)
            scores.append(score)
        return scores

    def compute_client_cosine_scores_with_avg(self, raw_client_grad_list):
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        cosine_scores = []
        for feature in importance_feature_list:
            score = 1-spatial.distance.cosine(feature, self.average)  # [0, 2]
            cosine_scores.append(score)
        # for i in range(0, num_client):
        #     dists = []
        #     for j in range(0, num_client):
        #         if i != j:
        #             dists.append(1-spatial.distance.cosine(importance_feature_list[i], importance_feature_list[j]))
        #     cosine_scores.append(sum(dists) / len(dists))
        return cosine_scores

    def compute_client_krum_scores(self, raw_client_grad_list):
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        return compute_krum_score(importance_feature_list,
                                         client_num_after_trim=math.floor(len(raw_client_grad_list) / 2), p=1)

    def compute_client_krum_cosine_scores(self, raw_client_grad_list):
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        krum_scores = compute_krum_score(importance_feature_list,
                                         client_num_after_trim=math.floor(len(raw_client_grad_list) / 2), p=1)

        cosine_scores = self.compute_client_cosine_scores(raw_client_grad_list)
        normalized_krum_scores = krum_scores #self.get_normalized_scores(krum_scores)
        normalized_cosine_scores = cosine_scores # self.get_normalized_scores(cosine_scores)
        print(f"normalized_krum_scores = {normalized_krum_scores}")
        print(f"normalized_cosine_scores = {normalized_cosine_scores}")

        scores = [normalized_krum_scores[i] + 0 * normalized_cosine_scores[i] for i in range(len(raw_client_grad_list))]
        print(f"krum scores = {krum_scores}")
        print(f"cosine scores = {cosine_scores}")
        print(f"scores = {scores}")
        return scores

    def get_normalized_scores(self, scores):
        normalized_scores = []
        for score in scores:
            normalized_scores.append((score - np.min(scores)) / (np.max(scores) - np.min(scores)))
        return normalized_scores

    def get_transformed_scores(self, scores, high, low):
        transformed_scores = []
        for score in scores:
            transformed_scores.append(((score - np.min(scores)) / (np.max(scores) - np.min(scores)) ) * (high - low) + (high - low) / 2)
        return transformed_scores

    def compute_client_foolsgold_scores(self, raw_client_grad_list):
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        return FoolsGoldDefense.fools_gold_score(importance_feature_list).tolist()


    def compute_client_cosine_scores(self, raw_client_grad_list):
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        cosine_scores = []
        num_client = len(importance_feature_list)
        for i in range(0, num_client):
            dists = []
            for j in range(0, num_client):
                if i != j:
                    dists.append(1-spatial.distance.cosine(importance_feature_list[i], importance_feature_list[j]))
            cosine_scores.append(sum(dists) / len(dists))
        return cosine_scores



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
