import numpy as np
import torch
from .base_dp_mechanism import BaseDPMechanism
from ..common.utils import check_params


class Gaussian(BaseDPMechanism):
    def __init__(self, args):
        if hasattr(args, "sigma") and isinstance(args.sigma, float):
            self._scale = args.sigma
        elif hasattr(args, "epsilon") and hasattr(args, "delta") and hasattr(args, "sensitivity"):
            check_params(args.epsilon, args.delta, args.sensitivity)
            if args.epsilon == 0 or args.delta == 0:
                raise ValueError("Neither Epsilon nor Delta can be zero")
            if args.epsilon > 1.0:
                raise ValueError(
                    "Epsilon cannot be greater than 1. "
                )
            self._scale = (
                np.sqrt(2 * np.log(1.25 / float(args.delta)))
                * float(args.sensitivity)
                / float(args.epsilon)
            )
        else:
            raise ValueError("Missing necessary parameters for Gaussian Mechanism")

    @classmethod
    def add_noise_using_sigma(cls, sigma, size):
        if not isinstance(sigma, float):
            raise ValueError("sigma should be a float")
        return torch.normal(mean=0, std=sigma, size=size)

    def compute_noise(self, size):
        return torch.normal(mean=0, std=self._scale, size=size)

    # def clip_gradients(self, grad): # Gaussian: 2 norm
    #     new_grad = dict()
    #     for k in grad.keys():
    #         new_grad[k] = max(1, grad[k].norm(2)) / self.clipping
    #     return new_grad
