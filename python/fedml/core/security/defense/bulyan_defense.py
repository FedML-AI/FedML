from typing import Callable, List, Tuple, Any
import numpy as np
from .defense_base import BaseDefenseMethod
from ..common.utils import vectorize_weight, is_weight_param
from collections import defaultdict, OrderedDict
import torch

"""
defense @ server, added by Chulin, 07/10/2022
"The Hidden Vulnerability of Distributed Learning in Byzantium. "
http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf

Bulyan(A) requires n ≥ 4f + 3 received gradients in two steps.
Steps: 
(1) recursively use A (e.g., Krum) to select θ = n − 2f gradients,
(2) with θ = n−2f gradients, generate the resulting gradient G, where each i-th coordinate of G is equal to the
    average of the β closest i-th coordinates to the median i-th coordinate of the θ selected gradients.

With the aggregated gradient, the parameter server performs a gradient descent update.
"""


class BulyanDefense(BaseDefenseMethod):
    """
    Bulyan Defense for Federated Learning.

    Bulyan Defense is a defense method for federated learning that aims to mitigate the impact of Byzantine clients
    by selecting a subset of clients' gradients for aggregation.

    Args:
        config: Configuration parameters for the defense.
            - byzantine_client_num (int): The number of Byzantine (malicious) clients.
            - client_num_per_round (int): The total number of clients participating in each aggregation round.

    Attributes:
        byzantine_client_num (int): The number of Byzantine (malicious) clients.
        client_num_per_round (int): The total number of clients participating in each aggregation round.
    """
    def __init__(self, config):
        self.byzantine_client_num = config.byzantine_client_num
        self.client_num_per_round = config.client_num_per_round

        assert self.client_num_per_round >= 4 * self.byzantine_client_num + 3, (
            "users_count>=4*corrupted_count + 3",
            self.client_num_per_round,
            self.byzantine_client_num,
        )

    def run(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ) -> OrderedDict:
        """
        Run the Bulyan Defense to aggregate gradients.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]): A list of tuples containing the number of samples
                and gradients for each client.
            base_aggregation_func (Callable, optional): The base aggregation function to use. Default is None.
            extra_auxiliary_info (Any, optional): Additional auxiliary information. Default is None.

        Returns:
            OrderedDict: The aggregated gradients after applying the Bulyan Defense.
        """
        # note: raw_client_grad_list is a list, each item is (sample_num, gradients).
        num_clients = len(raw_client_grad_list)
        (num0, localw0) = raw_client_grad_list[0]
        local_params_len = vectorize_weight(localw0).shape[
            0
        ]  # lens of the flatted gradients

        _params = np.zeros((num_clients, local_params_len))
        for i in range(num_clients):
            _params[i] = (
                vectorize_weight(raw_client_grad_list[i][1]).cpu().detach().numpy()
            )

        select_indexs, selected_set, agg_grads = self._bulyan(
            _params, self.client_num_per_round, self.byzantine_client_num
        )

        aggregated_params = {}
        index_bias = 0

        for item_index, (k, v) in enumerate(localw0.items()):
            if is_weight_param(k):
                aggregated_params[k] = torch.from_numpy(
                    agg_grads[index_bias : index_bias + v.numel()]
                ).view(
                    v.size()
                )  # todo: gpu/cpu issue for torch
                index_bias += v.numel()
            else:
                aggregated_params[k] = v
        return aggregated_params

    def _bulyan(self, users_params, users_count, corrupted_count):
        """
        Perform the Bulyan aggregation.

        Args:
            users_params (numpy.ndarray): Gradients of users' parameters.
            users_count (int): The total number of users.
            corrupted_count (int): The number of corrupted (Byzantine) users.

        Returns:
            Tuple[List[int], List[numpy.ndarray], numpy.ndarray]: A tuple containing the selected indices,
                selected set of gradients, and the aggregated gradients.
        """
        assert users_count >= 4 * corrupted_count + 3
        set_size = users_count - 2 * corrupted_count
        selection_set = []
        select_indexs = []
        distances = self._krum_create_distances(users_params)

        while len(selection_set) < set_size:
            currently_selected = self._krum(
                users_params,
                users_count - len(selection_set),
                corrupted_count,
                distances,
                return_index=True,
            )

            selection_set.append(users_params[currently_selected])
            select_indexs.append(currently_selected)
            # remove the selected from next iterations:
            distances.pop(currently_selected)
            for remaining_user in distances.keys():
                distances[remaining_user].pop(currently_selected)

        agg_grads = self.trimmed_mean(selection_set, 2 * corrupted_count)

        return select_indexs, selection_set, agg_grads

    @staticmethod
    def trimmed_mean(users_params, corrupted_count):
        """
        Compute the trimmed mean of users' gradients.

        Args:
            users_params (numpy.ndarray): Gradients of users' parameters.
            corrupted_count (int): The number of corrupted (Byzantine) users.

        Returns:
            numpy.ndarray: The trimmed mean of gradients.
        """

        users_params = np.array(users_params)
        number_to_consider = int(users_params.shape[0] - corrupted_count) - 1
        current_grads = np.empty((users_params.shape[1],), users_params.dtype)

        for i, param_across_users in enumerate(users_params.T):
            med = np.median(param_across_users)
            good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[
                :number_to_consider
            ]
            current_grads[i] = np.mean(good_vals) + med

        return current_grads

    def _krum(
        self,
        users_params,
        users_count,
        corrupted_count,
        distances=None,
        return_index=False,
    ):
        """
        Perform the Krum selection.

        Args:
            users_params (numpy.ndarray): Gradients of users' parameters.
            users_count (int): The total number of users.
            corrupted_count (int): The number of corrupted (Byzantine) users.
            distances (dict, optional): Precomputed distances between users. Default is None.
            return_index (bool, optional): Whether to return the selected index. Default is False.

        Returns:
            numpy.ndarray or int: The selected gradients or index.
        """

        non_malicious_count = users_count - corrupted_count
        minimal_error = 1e20
        minimal_error_index = -1

        if distances is None:
            distances = self._krum_create_distances(users_params)
        for user in distances.keys():
            errors = sorted(distances[user].values())
            current_error = sum(errors[:non_malicious_count])
            if current_error < minimal_error:
                minimal_error = current_error
                minimal_error_index = user

        if return_index:
            return minimal_error_index
        else:
            return users_params[minimal_error_index]

    @staticmethod
    def _krum_create_distances(users_params):
        """
        Create pairwise distances between users' gradients.

        Args:
            users_params (numpy.ndarray): Gradients of users' parameters.

        Returns:
            dict: A dictionary containing pairwise distances between users' gradients.
        """
        distances = defaultdict(dict)
        for i in range(len(users_params)):
            for j in range(i):
                distances[i][j] = distances[j][i] = np.linalg.norm(
                    users_params[i] - users_params[j]
                )
        return distances
