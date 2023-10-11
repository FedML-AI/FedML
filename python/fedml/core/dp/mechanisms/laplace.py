import numpy as np
import torch
from .base_dp_mechanism import BaseDPMechanism
from ..common.utils import check_params


class Laplace(BaseDPMechanism):
    """
    The classical Laplace mechanism in differential privacy.

    Attributes:
        epsilon (float): The privacy parameter epsilon.
        delta (float): The privacy parameter delta (default is 0.0).
        sensitivity (float): The sensitivity of the mechanism (default is 1).

    Methods:
        __init__(self, epsilon, delta=0.0, sensitivity=1): Initialize the Laplace mechanism.
        compute_noise(self, size): Generate Laplace noise.
        get_rdp_scale(self): Get the RDP (Rényi Differential Privacy) scale of the mechanism.
    """

    def __init__(self, epsilon, delta=0.0, sensitivity=1):
        """
        Initialize the Laplace mechanism.

        Args:
            epsilon (float): The privacy parameter epsilon.
            delta (float, optional): The privacy parameter delta (default is 0.0).
            sensitivity (float, optional): The sensitivity of the mechanism (default is 1).
        """
        check_params(epsilon, delta, sensitivity)
        self.scale = float(sensitivity) / \
            (float(epsilon) - np.log(1 - float(delta)))
        self.sensitivity = sensitivity

    def compute_noise(self, size):
        """
        Generate Laplace noise.

        Args:
            size (int or tuple): The size of the noise vector.

        Returns:
            torch.Tensor: A tensor containing Laplace noise.
        """
        return torch.tensor(np.random.laplace(loc=0.0, scale=self.scale, size=size))

    def get_rdp_scale(self):
        """
        Get the RDP (Rényi Differential Privacy) scale of the mechanism.

        Returns:
            float: The RDP scale of the mechanism.
        """
        return self.scale / self.sensitivity
