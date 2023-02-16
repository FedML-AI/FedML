"""
Reference:
autodp: https://github.com/yuxiangw/autodp
Opacus: https://github.com/pytorch/opacus
"""
import logging
import numpy as np
from typing import List, Optional, Tuple, Union
from fedml.core.dp.budget_accountant.rdp_analysis import compute_rdp


def stable_logsumexp_two(x, y):
    a = np.maximum(x, y)
    # if np.isneginf(a):
    #    return a
    # else:
    return a + np.log(np.exp(x - a) + np.exp(y - a))


class RDP_Accountant:
    def __init__(self, dp_param, alpha, dp_mechanism="gaussian", args=None):
        self.alpha = alpha
        if dp_mechanism not in ["gaussian", "laplace"]:
            raise Exception(f"the DP mechanism is not supported: {dp_mechanism}")
        self.dp_mechanism = dp_mechanism

        self.noise_multiplier = args.sigma
        self.max_grad_norm = args.max_grad_norm
        self.clipping = args.clipping
        self.history = []
        self.iteration_num = 1

        # if dp_mechanism == "gaussian":
        #     self.rdp_epsilon = self.RDP_gaussian(sigma=dp_param)
        # elif dp_mechanism == "laplace":
        #     self.rdp_epsilon = self.RDP_laplace(rdp_scale=dp_param)
        # else:
        #     raise Exception(f"the DP mechanism is not supported: {dp_mechanism}")

    # def RDP_gaussian(self, sigma):
    #     """
    #     Args:
    #         sigma: normalized noise level: std divided by global L2 sensitivity
    #         alpha: The order of the Renyi Divergence
    #
    #     Return: Evaluation of the RDP's epsilon
    #     """
    #     assert (sigma > 0)
    #     assert (self.alpha >= 0)
    #     return 0.5 / sigma ** 2 * self.alpha

    def get_epsilon_laplace(self, rdp_scale):
        """
        Args:
            rdp_scale: the ratio of the scale parameter and L1 sensitivity
            alpha: The order of the Renyi Divergence
        Return: Evaluation of the RDP's epsilon
        """
        alpha = 1.0 * self.alpha
        if np.isinf(alpha):
            return 1 / rdp_scale
        elif alpha == 1:
            # KL-divergence
            return 1 / rdp_scale + np.exp(-1 / rdp_scale) - 1
        elif alpha > 1:  # alpha > 1
            return stable_logsumexp_two((alpha - 1.0) / rdp_scale + np.log(alpha / (2.0 * alpha - 1)),
                                        -1.0 * alpha / rdp_scale + np.log((alpha - 1.0) / (2.0 * alpha - 1))) / (
                               alpha - 1)
        elif alpha == 0.5:
            return -2 * (-1.0 / (2 * rdp_scale) + np.log(
                1 + 1.0 / (2 * rdp_scale)))  # -2*np.log(np.exp(-1.0/(2*b))*(1+1.0/(2*b)))
        else:
            return np.log(
                alpha / (2.0 * alpha - 1) * np.exp((alpha - 1.0) / rdp_scale) + (alpha - 1.0) / (
                            2.0 * alpha - 1) * np.exp(
                    -1.0 * alpha / rdp_scale)) / (alpha - 1)
            # Handling the case when alpha = 1/2?

    def get_epsilon(
            self, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ):
        if self.dp_mechanism == "gaussian":
            return self.get_epsilon_gaussian(delta, alphas)
        else:
            return self.get_epsilon_laplace(rdp_scale=None)  # todo:

    def get_epsilon_gaussian(
            self, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ):
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            alphas: List of RDP orders (alphas) used to search for the optimal conversion
                between RDP and (epd, delta)-DP
        """
        if not self.history:
            return 0, 0

        if alphas is None:
            alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        rdp = sum(
            [
                compute_rdp(
                    q=sample_rate,
                    noise_multiplier=noise_multiplier,
                    steps=num_steps,
                    orders=alphas,
                )
                for (noise_multiplier, sample_rate, num_steps) in self.history
            ]
        )
        eps, best_alpha = self.get_privacy_spent(
            orders=alphas, rdp=rdp, delta=delta
        )
        return float(eps)

    def get_privacy_spent(
            *, orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
    ) -> Tuple[float, float]:
        r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
        multiple RDP orders and target ``delta``.
        The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
        is based on the theorem presented in the following work:
        Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
        International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
        Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
        Args:
            orders: An array (or a scalar) of orders (alphas).
            rdp: A list (or a scalar) of RDP guarantees.
            delta: The target delta.
        Returns:
            Pair of epsilon and optimal order alpha.
        Raises:
            ValueError
                If the lengths of ``orders`` and ``rdp`` are not equal.
        """
        orders_vec = np.atleast_1d(orders)
        rdp_vec = np.atleast_1d(rdp)

        if len(orders_vec) != len(rdp_vec):
            raise ValueError(
                f"Input lists must have the same length.\n"
                f"\torders_vec = {orders_vec}\n"
                f"\trdp_vec = {rdp_vec}\n"
            )

        eps = (
                rdp_vec
                - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
                + np.log((orders_vec - 1) / orders_vec)
        )

        # special case when there is no privacy
        if np.isnan(eps).all():
            return np.inf, np.nan

        idx_opt = np.nanargmin(eps)  # Ignore NaNs
        if idx_opt == 0 or idx_opt == len(eps) - 1:
            extreme = "smallest" if idx_opt == 0 else "largest"
            logging.info(
                f"Optimal order is the {extreme} alpha. Please consider expanding the range of alphas to get a tighter privacy bound."
            )
        return eps[idx_opt], orders_vec[idx_opt]


    def step(self, *, noise_multiplier: float, sample_rate: float):
        if len(self.history) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.history.pop()
            if last_noise_multiplier == noise_multiplier and last_sample_rate == sample_rate:
                self.history.append((last_noise_multiplier, last_sample_rate, num_steps + 1))
            else:
                self.history.append((last_noise_multiplier, last_sample_rate, num_steps))
                self.history.append((noise_multiplier, sample_rate, 1))

        else:
            self.history.append((noise_multiplier, sample_rate, 1))
