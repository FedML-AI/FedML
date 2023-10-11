import logging
import math
from collections import OrderedDict
import numpy as np
from scipy import spatial
from .defense_base import BaseDefenseMethod
from typing import List, Tuple, Dict, Any
from ..common.utils import (
    compute_euclidean_distance,
    compute_middle_point,
    compute_krum_score, compute_gaussian_distribution,
)
import torch


class ThreeSigmaKrumDefense(BaseDefenseMethod):
    """
    Three-Sigma Defense with Krum-based Malicious Client Detection for Federated Learning.

    This defense method performs a Three-Sigma-based defense with Krum-based malicious client detection for federated learning.

    Args:
        config: Configuration object for defense parameters.

    Methods:
        defend_before_aggregation(
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            extra_auxiliary_info: Any = None,
        ) -> List[Tuple[float, OrderedDict]]:
        Perform defense before aggregation.

        kick_out_poisoned_local_models(
            client_scores: List[float],
            raw_client_grad_list: List[Tuple[float, OrderedDict]]
        ) -> Tuple[List[Tuple[float, OrderedDict]], List[float]]:
        Remove poisoned local models based on client scores.

        get_malicious_client_idxs() -> List[int]:
        Get indices of detected malicious clients.

        set_potential_malicious_clients(potential_malicious_client_idxs: List[int]):
        Set potential malicious client indices.

        compute_avg_with_krum(raw_client_grad_list: List[Tuple[float, OrderedDict]]) -> List[float]:
        Compute an average feature with Krum-based malicious client detection.

        compute_l2_scores(raw_client_grad_list: List[Tuple[float, OrderedDict]]) -> List[float]:
        Compute L2 scores for client models.

        compute_client_cosine_scores(raw_client_grad_list: List[Tuple[float, OrderedDict]]) -> List[float]:
        Compute cosine similarity scores between client models.

        _get_importance_feature(raw_client_grad_list: List[Tuple[float, OrderedDict]]) -> List[float]:
        Get importance features from raw client gradients.
    """
    def __init__(self, config):
        """
        Initialize the ThreeSigmaKrumDefense.

        Args:
            config: Configuration object for defense parameters.
        """
        self.average = None
        self.upper_bound = 0
        self.malicious_client_idxs = []
        # OutlierDetectionDefense will set this list; when it is empty, kick out detected malicious models directly
        self.potential_malicious_client_idxs = None
        if hasattr(config, "bound_param") and isinstance(config.bound_param, float):
            self.bound_param = config.bound_param
        else:
            self.bound_param = 1

    ###################### version 3: re-compute gaussian distribution each round
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
        if self.average is None:
            self.average = self.compute_avg_with_krum(raw_client_grad_list)
        client_scores = self.compute_l2_scores(raw_client_grad_list)
        mu, sigma = compute_gaussian_distribution(client_scores)
        self.upper_bound = mu + self.bound_param * sigma
        print(f"client socres = {client_scores}")
        print(f"mu = {mu}, sigma = {sigma}, upperbound = {self.upper_bound}")
        new_client_models, _ = self.kick_out_poisoned_local_models(
            client_scores, raw_client_grad_list
        )
        importance_feature_list = self._get_importance_feature(new_client_models)
        self.average = self.compute_an_average_feature(importance_feature_list)
        return new_client_models

    def compute_an_average_feature(self, importance_feature_list):
        """
        Remove poisoned local models based on client scores.

        Args:
            client_scores (List[float]): List of client scores.
            raw_client_grad_list (List[Tuple[float, OrderedDict]]):
                List of tuples containing client gradients as OrderedDict.

        Returns:
            Tuple[List[Tuple[float, OrderedDict]], List[float]]:
                Tuple containing gradient list after removing poisoned models
                and updated client scores.
        """
        alphas = [1 / len(importance_feature_list)] * len(importance_feature_list)
        return compute_middle_point(alphas, importance_feature_list)

    ##################### version 2: remove poisoned model scores in score list
    # def defend_before_aggregation(
    #     self,
    #     raw_client_grad_list: List[Tuple[float, OrderedDict]],
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
    #     raw_client_grad_list: List[Tuple[float, OrderedDict]],
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
        """
        Get indices of detected malicious clients.

        Returns:
            List[int]: List of indices of malicious clients.
        """

        print(f"upper bound = {self.upper_bound}")
        # traverse the score list in a reversed order
        self.malicious_client_idxs = []
        logging.info(f"potential_malicious_client_idxs = {self.potential_malicious_client_idxs}")
        for i in range(len(client_scores) - 1, -1, -1):
            if client_scores[i] > self.upper_bound:
                if self.potential_malicious_client_idxs is None or i in self.potential_malicious_client_idxs:
                    raw_client_grad_list.pop(i)
                    self.malicious_client_idxs.append(i)
                    logging.info(f"kick out -- {i}")
        return raw_client_grad_list, client_scores

    def get_malicious_client_idxs(self):
        """
        Set potential malicious client indices.

        Args:
            potential_malicious_client_idxs: List of potential malicious client indices.
        """
        return self.malicious_client_idxs

    def set_potential_malicious_clients(self, potential_malicious_client_idxs):
        """
        Compute an average feature with Krum-based malicious client detection.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]):
                List of tuples containing client gradients as OrderedDict.

        Returns:
            List[float]: List representing an average feature.
        """
        self.potential_malicious_client_idxs = None # potential_malicious_client_idxs todo

    def compute_avg_with_krum(self, raw_client_grad_list):
        """
        Compute L2 scores for client models.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]):
                List of tuples containing client gradients as OrderedDict.

        Returns:
            List[float]: List of L2 scores.
        """
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        krum_scores = compute_krum_score(
            importance_feature_list,
            client_num_after_trim=math.ceil(len(raw_client_grad_list) / 2) - 1,
        )
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: math.ceil(len(raw_client_grad_list) / 2) - 1]
        honest_importance_feature_list = [
            importance_feature_list[i] for i in score_index
        ]
        return self.compute_an_average_feature(honest_importance_feature_list)

    def compute_l2_scores(self, raw_client_grad_list):
        """
        Compute L2 scores for client models.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]):
                List of tuples containing client gradients as OrderedDict.

        Returns:
            List[float]: List of L2 scores.
        """
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        scores = []
        for feature in importance_feature_list:
            score = compute_euclidean_distance(torch.Tensor(feature), self.average)
            scores.append(score)
        return scores

    def compute_client_cosine_scores(self, raw_client_grad_list):
        """
        Compute cosine similarity scores between client models.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]):
                List of tuples containing client gradients as OrderedDict.

        Returns:
            List[float]: List of cosine similarity scores.
        """
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        cosine_scores = []
        num_client = len(importance_feature_list)
        for i in range(0, num_client):
            dists = []
            for j in range(0, num_client):
                if i != j:
                    dists.append(
                        1
                        - spatial.distance.cosine(
                            importance_feature_list[i], importance_feature_list[j]
                        )
                    )
            cosine_scores.append(sum(dists) / len(dists))
        return cosine_scores

    def _get_importance_feature(self, raw_client_grad_list):
        """
        Get importance features from raw client gradients.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]):
                List of tuples containing client gradients as OrderedDict.

        Returns:
            List[float]: List of importance feature vectors.
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
