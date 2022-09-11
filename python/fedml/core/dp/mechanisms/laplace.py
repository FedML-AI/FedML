import numpy as np
import torch
from .base_dp_mechanism import BaseDPMechanism
from ..common.utils import check_params


class Laplace(BaseDPMechanism):
    """
    The classical Laplace mechanism in differential privacy.
    """

    def __init__(self, epsilon, delta=0.0, sensitivity=1):
        check_params(epsilon, delta, sensitivity)
        self.scale = float(sensitivity) / (float(epsilon) - np.log(1 - float(delta)))

    def compute_noise(self, size):
        return torch.tensor(np.random.laplace(loc=0.0, scale=self.scale, size=size))

    # def clip_gradients(self, grad): # Laplace: 1 norm
    #     new_grad = dict()
    #     for k in grad.keys():
    #         new_grad[k] = max(1, grad[k].norm(1)) / self.clipping
    #     return new_grad

