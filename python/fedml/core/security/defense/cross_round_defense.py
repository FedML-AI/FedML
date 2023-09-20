import numpy as np
from scipy import spatial
from .defense_base import BaseDefenseMethod
from typing import Callable, List, Tuple, Dict, Any
from collections import OrderedDict


# check whether attack happens
# 1. Compare with global model, compute a similarity
# 2. Compare with local model in the last round, compute a similarity
#
# very little difference: lazy worker, kickout
# too much difference: malicious, need further defense
# todo: pretraining round?
class CrossRoundDefense(BaseDefenseMethod):
    """
    CrossRoundDefense for Federated Learning.

    CrossRoundDefense is a defense method for federated learning that detects potentially poisoned workers
    based on cosine similarity between client and global model features across training rounds.

    Args:
        config: Configuration parameters for the defense, including 'upperbound' and 'lowerbound'.

    Attributes:
        potentially_poisoned_worker_list (list): List of potentially poisoned worker indices.
        lazy_worker_list (list): List of lazy worker indices.
        upperbound (float): Threshold for detecting potential attacks.
        lowerbound (float): Threshold for defining "very limited difference."
        client_cache (list): Cache of client features for comparison across training rounds.
        training_round (int): The current training round.
        is_attack_existing (bool): Flag indicating whether an attack exists in the current round.
    """
    def __init__(self, config):
        self.potentially_poisoned_worker_list = []
        self.lazy_worker_list = None
        # cosine similarity in [0, 2] 0 means 2 vectors are same
        self.upperbound = 0.31  # cosine similarity > upperbound: attack may happen; need further defense
        self.lowerbound = 0.0000001  # cosine similarity < lowerbound is defined as ``very limited difference''-> lazy worker
        self.client_cache = None
        self.training_round = 1
        self.is_attack_existing = True  # for the first round, true

    def defend_before_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            extra_auxiliary_info: Any = None,
    ):
        """
        Apply CrossRoundDefense before model aggregation.

        Args:
            raw_client_grad_list (list): List of client gradients for the current round.
            extra_auxiliary_info: Global model or auxiliary information.

        Returns:
            list: List of potentially poisoned client gradients.
        """
        self.is_attack_existing = False
        client_features = self._get_importance_feature(raw_client_grad_list)
        if self.training_round == 1:
            self.training_round += 1
            self.client_cache = client_features
            return raw_client_grad_list
        self.lazy_worker_list = []
        self.potentially_poisoned_worker_list = []
        # extra_auxiliary_info: global model
        global_model_feature = self._get_importance_feature_of_a_model(
            extra_auxiliary_info
        )
        client_wise_scores, global_wise_scores = self.compute_client_cosine_scores(
            client_features, global_model_feature
        )
        print(f"client_wise_scores = {client_wise_scores}")
        print(f"global_wise_scores = {global_wise_scores}")

        for i in range(len(client_wise_scores)):
            # if (
            #         client_wise_scores[i] < self.lowerbound
            #         or global_wise_scores[i] < self.lowerbound
            # ):
            #     self.lazy_worker_list.append(i)  # will be directly kicked out later
            # el
            if (
                    client_wise_scores[i] > self.upperbound
                    or global_wise_scores[i] > self.upperbound
            ):
                self.is_attack_existing = True
                self.potentially_poisoned_worker_list.append(i)

        for i in range(len(client_features) - 1, -1, -1):
            # if i in self.lazy_worker_list:
            #     raw_client_grad_list.pop(i)
            if i not in self.potentially_poisoned_worker_list:
                self.client_cache[i] = client_features[i]
        self.training_round += 1
        print(f"self.potentially_poisoned_worker_list = {self.potentially_poisoned_worker_list}")
        print(f"self.lazy_worker_list = {self.lazy_worker_list}")
        return raw_client_grad_list

    def get_potential_poisoned_clients(self):
        """
        Get the list of potentially poisoned client indices.

        Returns:
            list: List of potentially poisoned client indices.
        """
        return self.potentially_poisoned_worker_list

    def compute_client_cosine_scores(self, client_features, global_model_feature):
        """
        Compute cosine similarity scores between client features and global model features.

        Args:
            client_features (list): List of client feature vectors.
            global_model_feature (list): Feature vector of the global model.

        Returns:
            tuple: Two lists of cosine similarity scores for each client (client-wise and global-wise).
        """
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
        """
        Extract importance features from client gradients.

        Args:
            raw_client_grad_list (list): List of client gradients.

        Returns:
            list: List of extracted importance feature vectors.
        """
        ret_feature_vector_list = []
        for idx in range(len(raw_client_grad_list)):
            raw_grad = raw_client_grad_list[idx]
            (p, grad) = raw_grad

            feature_vector = self._get_importance_feature_of_a_model(grad)
            ret_feature_vector_list.append(feature_vector)
        return ret_feature_vector_list

    @classmethod
    def _get_importance_feature_of_a_model(self, grad):
        """
        Extract importance feature from a client gradient.

        Args:
            grad (OrderedDict): Client gradient.

        Returns:
            numpy.ndarray: Importance feature vector.
        """
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
