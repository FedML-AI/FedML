from collections import OrderedDict
from typing import List, Tuple, Any
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
    """
    CClip Defense for Federated Learning.

    CClip (Coordinate-wise Clipping) Defense is a defense method for federated learning that clips gradients at each
    coordinate to mitigate the impact of Byzantine clients.

    Args:
        config: Configuration parameters for the defense.
            - tau (float, optional): The clipping radius. Default is 10.
            - bucket_size (int, optional): The number of elements in each bucket when partitioning gradients.
              Default is None.

    Attributes:
        tau (float): The clipping radius.
        bucket_size (int): The number of elements in each bucket when partitioning gradients.
        initial_guess (OrderedDict): The initial guess for the global model.
    """

    def __init__(self, config):
        self.config = config
        if hasattr(config, "tau") and type(config.tau) in [int, float] and config.tau > 0:
            # clipping raduis; tau = 10 / (1-beta), beta is the coefficient of momentum
            self.tau = config.tau
        else:
            self.tau = 10  # default: no momentum, beta = 0
        # element # in each bucket; a grad_list is partitioned into floor(len(grad_list)/bucket_size) buckets
        self.bucket_size = config.bucket_size
        self.initial_guess = None

    def defend_before_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            extra_auxiliary_info: Any = None,
    ):
        """
        Apply CClip Defense before aggregation.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]): A list of tuples containing the number of samples
                and gradients for each client.
            extra_auxiliary_info (Any, optional): Additional auxiliary information. Default is None.

        Returns:
            List[Tuple[float, OrderedDict]]: The modified gradients after applying CClip Defense.
        """

        client_grad_buckets = Bucket.bucketization(
            raw_client_grad_list, self.bucket_size
        )
        self.initial_guess = self._compute_an_initial_guess(
            client_grad_buckets)
        bucket_num = len(client_grad_buckets)
        vec_local_w = [
            (
                client_grad_buckets[i][0],
                utils.vectorize_weight(client_grad_buckets[i][1]),
            )
            for i in range(bucket_num)
        ]
        vec_refs = utils.vectorize_weight(self.initial_guess)
        cclip_score = self._compute_cclip_score(vec_local_w, vec_refs)
        new_grad_list = []
        for i in range(bucket_num):
            tuple = OrderedDict()
            sample_num, bucket_params = client_grad_buckets[i]
            for k in bucket_params.keys():
                tuple[k] = (bucket_params[k] -
                            self.initial_guess[k]) * cclip_score[i]
            new_grad_list.append((sample_num, tuple))
        return new_grad_list

    def defend_after_aggregation(self, global_model):
        """
        Apply CClip Defense after aggregation.

        Args:
            global_model (OrderedDict): The global model after aggregation.

        Returns:
            OrderedDict: The modified global model after applying CClip Defense.
        """

        for k in global_model.keys():
            global_model[k] = self.initial_guess[k] + global_model[k]
        return global_model

    @staticmethod
    def _compute_an_initial_guess(client_grad_list):
        """
        Compute an initial guess for the global model.

        Args:
            client_grad_list (List[Tuple[float, OrderedDict]]): A list of tuples containing the number of samples
                and gradients for each client.

        Returns:
            OrderedDict: The initial guess for the global model.
        """
        # randomly select a gradient as the initial guess
        return client_grad_list[np.random.randint(0, len(client_grad_list))][1]

    def _compute_cclip_score(self, local_w, refs):
        """
        Compute the CClip score for each local gradient.

        Args:
            local_w (List[Tuple[float, numpy.ndarray]]): A list of tuples containing the number of samples and
                vectorized local gradients.
            refs (numpy.ndarray): Vectorized reference gradient.

        Returns:
            List[float]: A list of CClip scores for each local gradient.
        """
        cclip_score = []
        num_client = len(local_w)
        for i in range(0, num_client):
            dist = utils.compute_euclidean_distance(
                local_w[i][1], refs).item() + 1e-8
            score = min(1, self.tau / dist)
            cclip_score.append(score)
        return cclip_score
