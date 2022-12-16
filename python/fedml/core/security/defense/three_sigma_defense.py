import math
from collections import OrderedDict
import numpy as np
from .defense_base import BaseDefenseMethod
from typing import Callable, List, Tuple, Dict, Any
from ..common import utils
from scipy import spatial

### Original paper: https://arxiv.org/pdf/2107.05252.pdf
# training: In each iteration, each client k splits its local dataset into batches of size B,
# and runs for E local epochs batched-gradient descent through the local dataset
# to obtain local model, and sends it to the server.
# 1. each client sends a masked model, and the sum of all musks is 0 【dp noise???】
# 2. use m-krum to remove outlier

# idea in new approach
# This approach does cross-client defense;
# for cross-round verification: --> contribution assesment; can refer to foolsgold (using memory for each client) or residual_based_reweighting defense
# 1. use bucketing to eliminate effects of malicious clients
# 2. m-Krum requires to identify the # of malicious clients; it is not practical.
#    Two ways to solve this: 1) use some ways to compute a score (e.g., geometric, krum, foolsgold score, etc) for each client;
#                               Use the scores to compute a normal distribution.
#                               (variant: compute a distribution in each iteration, or compute a distribution before iterations)
#                               (variant 2: pre-process some iterations and obtain a distribution; adjust distribution later)
#                               client models with scores that are outliers in this distribution are removed (using 3sigma rule)
#    foolsgold use memory to compute a score. shortback: 1) all clients have to participant in computing 2) accuracy loss even all the clients are honest


# solutions:
# 1. compute a distribution
# 2. for each iteration, 1) compute scores for each client model;
#                        2) remove potential poisoned client models;
#                        3) do a bucketing over the remaining client models to mitigate potential poisoned effects of remained clients;
#                        4) do aggregation
# -------novelty ------: propose a method to detect poisoned client model
# to double_check: will doing bucketing mitigate poisoned effects of clients?


from ..common.bucket import Bucket
from ..common.utils import compute_geometric_median


class ThreeSigmaDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.memory = None
        self.iteration_num = 1
        self.score_list = []

        if hasattr(config, "bucketing_batch_size") and isinstance(config.bucketing_batch_size, int):
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
        self.bound_param = 2  # values outside mu +- 2sigma are outliers

        if hasattr(config, "to_keep_higher_scores") and isinstance(config.to_keep_higher_scores, bool):
            self.to_keep_higher_scores = config.to_keep_higher_scores
        else:
            self.to_keep_higher_scores = True  # true or false, depending on the score algo
        self.score_function = "foolsgold"

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        # grad_list = [grad for (_, grad) in raw_client_grad_list]
        client_scores = self.compute_client_scores(raw_client_grad_list)
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

    # def defend_on_aggregation(
    #     self,
    #     raw_client_grad_list: List[Tuple[float, OrderedDict]],
    #     base_aggregation_func: Callable = None,
    #     extra_auxiliary_info: Any = None,
    # ):  # raw_client_grad_list: batch_grad_list
    #     # ----------- geometric median part, or just use base_aggregation_func -------------
    #     # todo: why geometric median? what about other approaches?
    #     (num0, avg_params) = raw_client_grad_list[0]
    #     alphas = {alpha for (alpha, params) in raw_client_grad_list}
    #     alphas = {alpha / sum(alphas, 0.0) for alpha in alphas}
    #     for k in avg_params.keys():
    #         batch_grads = [params[k] for (alpha, params) in raw_client_grad_list]
    #         avg_params[k] = compute_geometric_median(alphas, batch_grads)
    #     return avg_params

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
        if self.score_function == "foolsgold":
            importance_feature_list = self._get_importance_feature(raw_client_grad_list)
            if self.memory is None:
                self.memory = importance_feature_list
            else:  # memory: potential bugs: grads in different iterations may be from different clients
                for i in range(len(raw_client_grad_list)):
                    self.memory[i] += importance_feature_list[i]
            return self.fools_gold_score(self.memory)

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

    @staticmethod
    def fools_gold_score(feature_vec_list):
        n_clients = len(feature_vec_list)
        cs = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(n_clients):
                cs[i][j] = 1 - spatial.distance.cosine(
                    feature_vec_list[i], feature_vec_list[j]
                )
        cs -= np.eye(n_clients)
        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        alpha = 1 - (np.max(cs, axis=1))
        alpha[alpha > 1.0] = 1.0
        alpha[alpha <= 0.0] = 1e-15

        # Rescale so that max value is alpha
         # print(np.max(alpha))
        alpha = alpha / np.max(alpha)
        alpha[(alpha == 1.0)] = 0.999999

        # Logit function
        alpha = np.log(alpha / (1 - alpha)) + 0.5
        # alpha[(np.isinf(alpha) + alpha > 1)] = 1
        # alpha[(alpha < 0)] = 0

        print("alpha = {}".format(alpha))

        return alpha
