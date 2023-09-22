import numpy as np
import torch
from .base_dp_mechanism import BaseDPMechanism
from ..common.utils import check_params


class Gaussian(BaseDPMechanism):
    """
    The Gaussian mechanism in differential privacy.

    Attributes:
        epsilon (float): The privacy parameter epsilon.
        delta (float): The privacy parameter delta (default is 0.0).
        sensitivity (float): The sensitivity of the mechanism (default is 1).

    Methods:
        __init__(self, epsilon, delta=0.0, sensitivity=1): Initialize the Gaussian mechanism.
        compute_noise(self, size): Generate Gaussian noise.
        compute_noise_using_sigma(cls, sigma, size): Generate Gaussian noise with a given standard deviation.
        get_rdp_scale(self): Get the RDP (Rényi Differential Privacy) scale of the mechanism.
    """

    def __init__(self, epsilon, delta=0.0, sensitivity=1):
        """
        Initialize the Gaussian mechanism.

        Args:
            epsilon (float): The privacy parameter epsilon.
            delta (float, optional): The privacy parameter delta (default is 0.0).
            sensitivity (float, optional): The sensitivity of the mechanism (default is 1).
        """
        check_params(epsilon, delta, sensitivity)
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")
        if epsilon > 1.0:
            raise ValueError(
                "Epsilon cannot be greater than 1. "
            )

        self.scale = (
            np.sqrt(2 * np.log(1.25 / float(delta)))
            * float(sensitivity)
            / float(epsilon)
        )

    @classmethod
    def compute_noise_using_sigma(cls, sigma, size):
        """
        Generate Gaussian noise with a given standard deviation.

        Args:
            sigma (float): The standard deviation of the Gaussian noise.
            size (int or tuple): The size of the noise vector.

        Returns:
            torch.Tensor: A tensor containing Gaussian noise.
        """
        if not isinstance(sigma, float):
            raise ValueError("sigma should be a float")
        return torch.normal(mean=0, std=sigma, size=size)

    def compute_noise(self, size):
        """
        Generate Gaussian noise.

        Args:
            size (int or tuple): The size of the noise vector.

        Returns:
            torch.Tensor: A tensor containing Gaussian noise.
        """
        return torch.normal(mean=0, std=self.scale, size=size)

    def get_rdp_scale(self):
        """
        Get the RDP (Rényi Differential Privacy) scale of the mechanism.

        Returns:
            float: The RDP scale of the mechanism.
        """
        return self.scale
