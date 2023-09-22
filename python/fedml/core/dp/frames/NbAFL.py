import math
from collections import OrderedDict
import torch
import numpy as np
from fedml.core.dp.frames.base_dp_solution import BaseDPFrame
from fedml.core.dp.mechanisms import Gaussian
from typing import List, Tuple
from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism

"""Federated Learning with Differential Privacy: Algorithms and Performance Analysis
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9069945"""


class NbAFL_DP(BaseDPFrame):
    """
    Non-Blocking Asynchronous Federated Learning with Differential Privacy Mechanism.

    Attributes:
        args: A namespace containing the configuration arguments for the mechanism.
        big_C_clipping (float): A clipping threshold for bounding model weights.
        total_round_num (int): The total number of communication rounds.
        small_c_constant (float): A constant used in the mechanism.
        client_num_per_round (int): The number of clients participating in each round.
        client_num_in_total (int): The total number of clients.
        epsilon (float): The privacy parameter epsilon.
        m (int): The minimum size of local datasets.

    Methods:
        __init__(self, args): Initialize the NbAFL_DP mechanism.
        add_local_noise(self, local_grad: OrderedDict): Add local noise to the gradients.
        add_global_noise(self, global_model: OrderedDict): Add global noise to the global model.
        set_params_for_dp(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]): Set parameters for DP.
    """

    def __init__(self, args):
        """
        Initialize the NbAFL_DP mechanism.

        Args:
            args: A namespace containing the configuration arguments for the mechanism.
        """
        super().__init__(args)
        self.set_ldp(
            DPMechanism(
                "gaussian",
                args.epsilon,
                args.delta
            )
        )
        """
        In the experiments, the authors chose C by taking the median of the norms of the unclipped parameters.
        This is not practical in reality. The server cannot obtain unclipped plaintext parameters. It can only
        get noised clipped parameters. So here we set C as a parameter indicated by users.
        """
        self.big_C_clipping = args.C  # C: a clipping threshold for bounding w_i
        self.total_round_num = args.comm_round  # T in the paper
        self.small_c_constant = np.sqrt(
            2 * math.log(1.25 / args.delta))  # the author indicated c >= sqrt(2ln(1.25/delta)
        self.client_num_per_round = args.client_num_per_round  # L in the paper
        self.client_num_in_total = args.client_num_in_total  # N in the paper
        self.epsilon = args.epsilon  # 0 < epsilon < 1
        """ The author said ''m is the minimum size of the local datasets''.
        In their paper, clients did not sample local data for training;
        In our setting, we set m to the minimum sample num of each round."""
        self.m = 0  # the minimum size of the local datasets

    def add_local_noise(self, local_grad: OrderedDict):
        """
        Add local noise to the gradients.

        Args:
            local_grad (OrderedDict): Local gradients.

        Returns:
            OrderedDict: Local gradients with added noise.
        """
        for k in local_grad.keys():  # Clip weight
            local_grad[k] = local_grad[k] / torch.max(torch.ones(size=local_grad[k].shape),
                                                      torch.abs(local_grad[k]) / self.big_C_clipping)
        return super().add_local_noise(local_grad=local_grad)

    def add_global_noise(self, global_model: OrderedDict):
        """
        Add global noise to the global model.

        Args:
            global_model (OrderedDict): Global model parameters.

        Returns:
            OrderedDict: Global model parameters with added noise.
        """
        if self.total_round_num > np.sqrt(self.client_num_in_total) * self.client_num_per_round:
            scale_d = 2 * self.small_c_constant * self.big_C_clipping * np.sqrt(
                np.power(self.total_round_num, 2) -
                np.power(self.client_num_per_round, 2) * self.client_num_in_total) / (
                self.m * self.client_num_in_total * self.epsilon)
            for k in global_model.keys():
                global_model[k] = Gaussian.compute_noise_using_sigma(
                    scale_d, global_model[k].shape)
        return global_model

    def set_params_for_dp(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        """
        Set parameters for Differential Privacy.

        Args:
            raw_client_model_or_grad_list (List[Tuple[float, OrderedDict]]): List of tuples containing sample numbers and gradients/models.
        """
        smallest_sample_num, _ = raw_client_model_or_grad_list[0]
        for (sample_num, _) in raw_client_model_or_grad_list:
            if smallest_sample_num > sample_num:
                smallest_sample_num = sample_num
        self.m = smallest_sample_num
