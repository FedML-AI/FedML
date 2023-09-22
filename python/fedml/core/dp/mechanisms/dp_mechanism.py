from .gaussian import Gaussian
from .laplace import Laplace
from fedml.core.dp.mechanisms import Gaussian, Laplace
import torch
from typing import Union, Iterable
from collections import OrderedDict

"""call dp mechanisms, e.g., Gaussian, Laplace """

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


class DPMechanism:
    """
    A class representing a Differential Privacy Mechanism.

    Attributes:
        mechanism_type (str): The type of differential privacy mechanism ('laplace' or 'gaussian').
        epsilon (float): The privacy parameter epsilon.
        delta (float): The privacy parameter delta.
        sensitivity (float, optional): The sensitivity of the mechanism (default is 1).

    Methods:
        __init__(self, mechanism_type, epsilon, delta, sensitivity=1): Initialize the DP mechanism.
        add_noise(self, grad): Add noise to a gradient.
        _compute_new_grad(self, grad): Compute a new gradient by adding noise.
        add_a_noise_to_local_data(self, local_data): Add noise to local data.
        get_rdp_scale(self): Get the RDP (Rényi Differential Privacy) scale of the mechanism.
    """

    def __init__(self, mechanism_type, epsilon, delta, sensitivity=1):
        """
        Initialize the Differential Privacy Mechanism.

        Args:
            mechanism_type (str): The type of differential privacy mechanism ('laplace' or 'gaussian').
            epsilon (float): The privacy parameter epsilon.
            delta (float): The privacy parameter delta.
            sensitivity (float, optional): The sensitivity of the mechanism (default is 1).
        """
        mechanism_type = mechanism_type.lower()
        if mechanism_type == "laplace":
            self.dp = Laplace(
                epsilon=epsilon, delta=delta, sensitivity=sensitivity
            )
        elif mechanism_type == "gaussian":
            self.dp = Gaussian(epsilon, delta=delta, sensitivity=sensitivity)
        else:
            raise NotImplementedError("DP mechanism not implemented!")

    def add_noise(self, grad):
        """
        Add noise to a gradient.

        Args:
            grad (OrderedDict): The gradient to which noise will be added.

        Returns:
            OrderedDict: A new gradient with added noise.
        """
        new_grad = OrderedDict()
        for k in grad.keys():
            new_grad[k] = self._compute_new_grad(grad[k])
        return new_grad

    def _compute_new_grad(self, grad):
        """
        Compute a new gradient by adding noise.

        Args:
            grad (torch.Tensor): The gradient tensor.

        Returns:
            torch.Tensor: A new gradient tensor with added noise.
        """
        noise = self.dp.compute_noise(grad.shape)
        return noise + grad

    def add_a_noise_to_local_data(self, local_data):
        """
        Add noise to local data.

        Args:
            local_data (list of tuples): Local data where each tuple represents a data point.

        Returns:
            list of tuples: Local data with added noise.
        """
        new_data = []
        for i in range(len(local_data)):
            data_tuple = []
            for x in local_data[i]:
                noisy_data = self._compute_new_grad(x)
                data_tuple.append(noisy_data)
            new_data.append(tuple(data_tuple))
        return new_data

    def get_rdp_scale(self):
        """
        Get the RDP (Rényi Differential Privacy) scale of the mechanism.

        Returns:
            float: The RDP scale of the mechanism.
        """
        return self.dp.get_rdp_scale()
