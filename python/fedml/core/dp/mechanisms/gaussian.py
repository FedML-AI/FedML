import numpy as np
import torch
from .base_dp_mechanism import BaseDPMechanism
from ..common.utils import check_params


class Gaussian(BaseDPMechanism):
    r"""The Gaussian mechanism in differential privacy.
    Some codes refer to IBM DP Library: https://github.com/IBM/differential-privacy-library
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

    def variance(self, value):
        return self._scale**2

    def compute_noise(self, size):
        return torch.normal(mean=0, std=self._scale, size=size)
