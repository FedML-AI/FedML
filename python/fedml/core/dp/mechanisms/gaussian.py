import numpy as np
import torch
from .base_dp_mechanism import BaseDPMechanism
from ..common.utils import check_params


class Gaussian(BaseDPMechanism):
    def __init__(self, epsilon=None, delta=0.0, sensitivity=1, dp_type='cdp', clipping_norm=None, args=None):
        # if hasattr(args, "epsilon") and hasattr(args, "delta") and hasattr(args, "sensitivity"):
        check_params(epsilon, delta, sensitivity)
        self.dp_type = dp_type
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")
        if epsilon > 1.0:
            raise ValueError(
                "Epsilon cannot be greater than 1. "
            )
        if dp_type == 'cdp':
            self._scale = (
                    args.clipping_norm * args.noise_multiplier
            )
        else:
            self._scale = (
                    np.sqrt(2 * np.log(1.25 / float(delta)))
                    * float(sensitivity)
                    / float(epsilon)
            )
        # else:
        #     raise ValueError("Missing necessary parameters for Gaussian Mechanism")

    @classmethod
    def compute_noise_using_sigma(cls, sigma, size):
        if not isinstance(sigma, float):
            raise ValueError("sigma should be a float")
        return torch.normal(mean=0, std=sigma, size=size)

    def compute_noise(self, size, qw):
        self._scale = self._scale / qw
        return torch.normal(mean=0, std=self._scale, size=size)

    # def clip_gradients(self, grad): # Gaussian: 2 norm
    #     new_grad = dict()
    #     for k in grad.keys():
    #         new_grad[k] = max(1, grad[k].norm(2)) / self.clipping
    #     return new_grad
