import numpy as np
from scipy import spatial
from .defense_base import BaseDefenseMethod
from typing import Callable, List, Tuple, Dict, Any
from collections import OrderedDict
from ..common.utils import (
    compute_euclidean_distance,
    # get_importance_feature,
    compute_middle_point,
    compute_krum_score, compute_gaussian_distribution,
)
import torch
import math


# check whether attack happens
# 1. Compare with global model, compute a similarity
# 2. Compare with local model in the last round, compute a similarity
#
# very little difference: lazy worker, kickout
# too much difference: malicious, need further defense
# todo: pretraining round?
class CrossRoundDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.potentially_poisoned_worker_list = []
        self.lazy_worker_list = None

        # self.upperbound = 1  # cosine similarity > upperbound: ``very limited difference''-> lazy worker
        self.lowerbound = config.cosine_similarity_bound  # cosine similarity < lowerbound attack may happen; need further defense
        self.client_cache = dict()
        self.training_round = 1
        self.is_attack_existing = True  # for the first round, true
        self.temp_client_features = None
        self.global_model_feature = None
        self.total_client_num = -1
        self.zero_reference = None
        self.upperbound = 1  # 0.999999

    def defend_before_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            extra_auxiliary_info: Any = None,
    ):
        self.temp_client_features = self._get_importance_feature(raw_client_grad_list)
        if self.training_round == 1:  # set attack exists by default for the first round and leave for second phase
            self.training_round += 1
            self.total_client_num = len(raw_client_grad_list)
            # self.client_cache = self.temp_client_features
            self.potentially_poisoned_worker_list = range(self.total_client_num)
            # Create a new vector with the same shape as feature_vector but with all weights being zero
            self.zero_reference = np.zeros(self.temp_client_features[0].shape)
            return raw_client_grad_list
        self.is_attack_existing = False

        self.lazy_worker_list = []
        self.potentially_poisoned_worker_list = []
        # extra_auxiliary_info: global model
        self.global_model_feature = self._get_importance_feature_of_a_model(
            extra_auxiliary_info
        )

        if self.training_round == 2:
            for i in range(self.total_client_num):
                if i not in self.client_cache:
                    self.client_cache[i] = self.global_model_feature
        client_wise_scores, global_wise_scores, zero_wise_scores = self.compute_client_cosine_scores(
            client_features=self.temp_client_features, global_model_feature=self.global_model_feature,
            zero_reference=self.zero_reference
        )

        for i in range(len(client_wise_scores)):
            # if (
            #         client_wise_scores[i] < self.lowerbound
            #         or global_wise_scores[i] < self.lowerbound
            # ):
            #     self.lazy_worker_list.append(i)  # will be directly kicked out later
            if client_wise_scores[i] < self.lowerbound or global_wise_scores[i] < self.lowerbound:
                self.is_attack_existing = True
                self.potentially_poisoned_worker_list.append(i)

        # for i in range(len(self.temp_client_features) - 1, -1, -1):
        #     # if i in self.lazy_worker_list:
        #     #     raw_client_grad_list.pop(i)
        #     if i not in self.potentially_poisoned_worker_list:
        #         self.client_cache[i] = self.temp_client_features[i]
        self.training_round += 1
        print(
            f"!!!!!!!!!!!!!!!!!!!!first phase: self.potentially_poisoned_worker_list = {self.potentially_poisoned_worker_list}")
        return raw_client_grad_list

    # def compute_gaussian_distribution(score_list):
    #     n = len(score_list)
    #     mu = sum(list(score_list)) / n
    #     temp = 0

    #     for i in range(len(score_list)):
    #         temp = (((score_list[i] - mu) ** 2) / (n - 1)) + temp
    #     sigma = math.sqrt(temp)
    #     return mu, sigma

    def compute_l2_scores(self, importance_feature_list):
        client_wise_distance_scores = []
        global_wise_distance_scores = []
        for i in range(len(importance_feature_list)):
            client_wise_distance_score = compute_euclidean_distance(torch.Tensor(importance_feature_list[i]),
                                                                    self.client_cache[i])
            global_wise_distance_score = compute_euclidean_distance(torch.Tensor(importance_feature_list[i]),
                                                                    self.global_model_feature)
            client_wise_distance_scores.append(client_wise_distance_score)
            global_wise_distance_scores.append(global_wise_distance_score)
        return client_wise_distance_scores, global_wise_distance_scores

    def renew_cache(self, real_poisoned_client_ids):
        for i in range(self.total_client_num):
            if i not in real_poisoned_client_ids:
                self.client_cache[i] = self.temp_client_features[i]
            else:
                if i not in self.client_cache and self.global_model_feature is not None:
                    self.client_cache[i] = self.global_model_feature

    def get_potential_poisoned_clients(self):
        return self.potentially_poisoned_worker_list

    def compute_client_cosine_scores(self, client_features, global_model_feature, zero_reference):
        client_wise_scores = []
        global_wise_scores = []
        zero_wise_scores = []
        num_client = len(client_features)
        for i in range(0, num_client):
            # spatial.distance.cosine ranges from 0 to 2; cosine_similarity below ranges from -1 to 1
            cosine_similarity = 1 - spatial.distance.cosine(client_features[i], self.client_cache[i])
            client_wise_scores.append(cosine_similarity)
            cosine_similarity = 1 - spatial.distance.cosine(client_features[i], global_model_feature)
            global_wise_scores.append(cosine_similarity)
            # cosine_similarity = 1 -  spatial.distance.cosine(client_features[i], zero_reference)
            cosine_similarity = 1 - spatial.distance.cosine(client_features[i], np.zeros(client_features[i].shape))
            # np.zeros(self.temp_client_features[0].shape)
            zero_wise_scores.append(cosine_similarity)
        return client_wise_scores, global_wise_scores, zero_wise_scores

    def _get_importance_feature(self, raw_client_grad_list):
        ret_feature_vector_list = []
        for idx in range(len(raw_client_grad_list)):
            raw_grad = raw_client_grad_list[idx]
            (p, grad) = raw_grad

            feature_vector = self._get_importance_feature_of_a_model(grad)
            ret_feature_vector_list.append(feature_vector)
        return ret_feature_vector_list

    @classmethod
    def _get_importance_feature_of_a_model(self, grad):
        # Get last key-value tuple
        (weight_name, importance_feature) = list(grad.items())[-2]
        # print(importance_feature)
        feature_len = np.array(
            importance_feature.cpu().data.detach().numpy().shape
        ).prod()
        feature_vector = np.reshape(
            importance_feature.cpu().data.detach().numpy(), feature_len
        )
        return feature_vector
