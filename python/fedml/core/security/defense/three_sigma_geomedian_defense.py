import math
from collections import OrderedDict
import numpy as np
from .defense_base import BaseDefenseMethod
from typing import Callable, List, Tuple, Dict, Any
from scipy import spatial
from ..common.utils import compute_geometric_median, compute_euclidean_distance
import torch


class ThreeSigmaGeoMedianDefense(BaseDefenseMethod):
    """
    Three-Sigma Defense with Geometric Median for Federated Learning.

    This defense method performs a Three-Sigma-based defense with geometric median for federated learning.

    Args:
        config: Configuration object for defense parameters.

    Methods:
        defend_before_aggregation(
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            extra_auxiliary_info: Any = None,
        ) -> List[Tuple[float, OrderedDict]]:
        Perform defense before aggregation.

        compute_gaussian_distribution() -> Tuple[float, float]:
        Compute the Gaussian distribution parameters.

        compute_client_scores(raw_client_grad_list) -> List[float]:
        Compute client scores.

        fools_gold_score(feature_vec_list) -> List[float]:
        Compute Fool's Gold scores.

        l2_scores(importance_feature_list) -> List[float]:
        Compute L2 scores.

    """

    def __init__(self, config):
        """
        Initialize the ThreeSigmaGeoMedianDefense.

        Args:
            config: Configuration object for defense parameters.
        """
        self.memory = None
        self.iteration_num = 1
        self.score_list = []
        self.geo_median = None

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

        if hasattr(config, "to_keep_higher_scores") and isinstance(config.to_keep_higher_scores, bool):
            self.to_keep_higher_scores = config.to_keep_higher_scores
        else:
            self.to_keep_higher_scores = False  # true or false, depending on the score algo
        self.score_function = "l2"

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        """
        Perform defense before aggregation.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]):
                List of tuples containing client gradients as OrderedDict.
            extra_auxiliary_info (Any, optional):
                Extra auxiliary information (currently unused).

        Returns:
            List[Tuple[float, OrderedDict]]: Gradient list after defense.
        """
        # grad_list = [grad for (_, grad) in raw_client_grad_list]
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
        return raw_client_grad_list

    def compute_gaussian_distribution(self):
        """
        Compute the Gaussian distribution parameters.

        Returns:
            Tuple[float, float]: Mean (mu) and standard deviation (sigma).
        """

        n = len(self.score_list)
        mu = sum(list(self.score_list)) / n
        temp = 0

        for i in range(len(self.score_list)):
            temp = (((self.score_list[i] - mu) ** 2) / (n - 1)) + temp
        sigma = math.sqrt(temp)
        print(f"mu = {mu}, sigma = {sigma}")
        return mu, sigma

    def compute_client_scores(self, raw_client_grad_list):
        """
        Compute client scores.

        Args:
            raw_client_grad_list: List of tuples containing client gradients as OrderedDict.

        Returns:
            List[float]: List of client scores.
        """
        importance_feature_list = self._get_importance_feature(
            raw_client_grad_list)
        if self.score_function == "foolsgold":
            if self.memory is None:
                self.memory = importance_feature_list
            else:  # memory: potential bugs: grads in different iterations may be from different clients
                for i in range(len(raw_client_grad_list)):
                    self.memory[i] += importance_feature_list[i]
            return self.fools_gold_score(self.memory)
        if self.score_function == "l2":
            if self.geo_median is None:
                # (num0, avg_params) = raw_client_grad_list[0]
                # alphas = {alpha for (alpha, params) in raw_client_grad_list}
                # alphas = {alpha / sum(alphas, 0.0) for alpha in alphas}
                alphas = [1/len(raw_client_grad_list)] * \
                    len(raw_client_grad_list)
                self.geo_median = compute_geometric_median(
                    alphas, importance_feature_list)
            return self.l2_scores(importance_feature_list)

    def l2_scores(self, importance_feature_list):
        """
        Compute L2 scores.

        Args:
            importance_feature_list: List of importance features.

        Returns:
            List[float]: List of L2 scores.
        """
        scores = []
        for feature in importance_feature_list:
            score = compute_euclidean_distance(
                torch.Tensor(feature), self.geo_median)
            scores.append(score)
        return scores

    def _get_importance_feature(self, raw_client_grad_list):
        """
        Get importance features for score computation.

        Args:
            raw_client_grad_list: List of tuples containing client gradients as OrderedDict.

        Returns:
            List[float]: List of importance features.
        """
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
        """
        Compute Fool's Gold scores.

        Args:
            feature_vec_list: List of importance features.

        Returns:
            List[float]: List of Fool's Gold scores.
        """
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
