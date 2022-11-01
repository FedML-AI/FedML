import math
import numpy as np
from scipy import spatial
from .defense_base import BaseDefenseMethod
from typing import Callable, List, Tuple, Dict, Any


# check whether attack happens
# 1. Compare with global model, compute a similarity
# 2. Compare with local model in the last round, compute a similarity
#
# very little difference: lazy worker, kickout
# too much difference: malicious, need further defense
# todo: pretraining round?
class CrossRoundDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.lazy_worker_list = None
        self.potential_malicious_client_idxs = []
        self.upperbound = 0.95  # cosine similarity > upperbound is defined as ``very limited difference''-> lazy worker
        self.lowerbound = 0.8  # cosine similarity < lowerbound: attack may happen; need further defense
        self.client_cache = None
        self.pretraining_round = 2
        self.potentially_poisoned_worker_list = None

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
        self.lazy_worker_list = []
        self.potentially_poisoned_worker_list = []
        # extra_auxiliary_info: global model
        global_model_feature = self._get_importance_feature_of_a_model(
            extra_auxiliary_info
        )
        client_features = self._get_importance_feature(raw_client_grad_list)
        if self.client_cache is None:
            self.client_cache = client_features
        client_wise_scores, global_wise_scores = self.compute_client_cosine_scores(
            client_features, global_model_feature
        )

        for i in range(len(client_wise_scores)):
            if (
                    client_wise_scores[i] > self.upperbound
                    or global_wise_scores[i] > self.upperbound
            ):
                self.lazy_worker_list.append(i)  # will be directly kicked out later
            elif (
                    client_wise_scores[i] < self.lowerbound
                    or global_wise_scores[i] < self.upperbound
            ):
                self.potentially_poisoned_worker_list.append(i)

        for i in range(len(client_features) - 1, -1, -1):
            if i in self.lazy_worker_list:
                raw_client_grad_list.pop(i)
            elif i not in self.potentially_poisoned_worker_list:
                self.client_cache[i] = client_features[i]
        return raw_client_grad_list

    def compute_client_cosine_scores(self, client_features, global_model_feature):
        client_wise_scores = []
        global_wise_scores = []
        num_client = len(client_features)
        for i in range(0, num_client):
            score = spatial.distance.cosine(client_features[i], self.client_cache[i])
            client_wise_scores.append(score)
            score = spatial.distance.cosine(client_features[i], global_model_feature)
            global_wise_scores.append(score)
        return client_wise_scores, global_wise_scores

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
