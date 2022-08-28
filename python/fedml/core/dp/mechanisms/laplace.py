import secrets
import numpy as np
import torch
from .base_dp_mechanism import BaseDPMechanism
from ..common.utils import check_params


class Laplace(BaseDPMechanism):
    """
    The classical Laplace mechanism in differential privacy.
    Some codes refer to IBM DP Library: https://github.com/IBM/differential-privacy-library
    """

    def __init__(self, *, epsilon, delta=0.0, sensitivity):
        check_params(epsilon, delta, sensitivity)
        self.scale = float(sensitivity) / (float(epsilon) - np.log(1 - float(delta)))
        self._rng = secrets.SystemRandom()

    def variance(self, value):
        return 2 * self.scale ** 2

    def compute_noise(self, size):
        return torch.tensor(np.random.laplace(loc=0.0, scale=self.scale, size=size))

