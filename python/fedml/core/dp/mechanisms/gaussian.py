"""
The classic Gaussian mechanism in differential privacy, and its derivatives.
"""
import secrets
import numpy as np
import torch

from ..common.utils import check_numeric_value, check_params


class Gaussian:
    r"""The Gaussian mechanism in differential privacy.
    This code refers to IBM DP Library: https://github.com/IBM/differential-privacy-library
    Our contribution: code refactoring; remove some redundant codes

    "The algorithmic foundations of differential privacy" [DR14]_."
    Samples from the Gaussian distribution are generated using two samples from `random.normalvariate` as in [HB21b]_,
    to prevent against reconstruction attacks due to limited floating point precision.

    Parameters
    ----------
    epsilon : float. Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, 1].  For ``epsilon > 1``, use
        :class:`.GaussianAnalytic`.

    delta : float. Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 1].

    sensitivity : float. The sensitivity of the mechanism.  Must be in [0, ∞).

    References
    ----------
    .. [DR14] Dwork, Cynthia, and Aaron Roth. "The algorithmic foundations of differential privacy."
    .. [HB21b] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in Differential Privacy."
    """

    def __init__(self, *, epsilon, delta, sensitivity):
        check_params(epsilon, delta, sensitivity)
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")
        if epsilon > 1.0:
            raise ValueError(
                "Epsilon cannot be greater than 1. If required, use GaussianAnalytic instead."
            )
        self._scale = (
            np.sqrt(2 * np.log(1.25 / float(delta)))
            * float(sensitivity)
            / float(epsilon)
        )
        # self._rng = secrets.SystemRandom()

    def bias(self, value):
        return 0.0

    def variance(self, value):
        return self._scale**2

    # def randomise(self, value):
    #     check_numeric_value(value)
    #     return value + self.compute_a_noise()

    # def compute_a_noise(self):
    #     standard_normal = (
    #         self._rng.normalvariate(0, 1) + self._rng.normalvariate(0, 1)
    #     ) / np.sqrt(2)
    #     return standard_normal * self._scale

    def compute_a_noise(self, size):
        print(f"scale = {self._scale}")
        return torch.normal(mean=0, std=self._scale, size=size)
