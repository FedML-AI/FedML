import numpy as np
import torch
from .base_dp_mechanism import BaseDPMechanism


class Gaussian(BaseDPMechanism):
    def __init__(self, *, epsilon, delta, sensitivity):
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")
        if epsilon > 1.0:
            raise ValueError(
                "Epsilon cannot be greater than 1. "
            )
        self._scale = (
            np.sqrt(2 * np.log(1.25 / float(delta)))
            * float(sensitivity)
            / float(epsilon)
        )

    def compute_noise(self, size):
        return torch.normal(mean=0, std=self._scale, size=size)

    # def clip_gradients(self, grad): # Gaussian: 2 norm
    #     new_grad = dict()
    #     for k in grad.keys():
    #         new_grad[k] = max(1, grad[k].norm(2)) / self.clipping
    #     return new_grad
