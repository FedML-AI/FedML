from fedml.core.dp.mechanisms import Gaussian, Laplace
import torch
from typing import Union, Iterable

from collections import OrderedDict

"""call dp mechanisms, e.g., Gaussian, Laplace """

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


class DPMechanism:
    def __init__(self, mechanism_type, epsilon, delta, sensitivity, args):
        mechanism_type = mechanism_type.lower()
        if mechanism_type == "laplace":
            self.dp = Laplace(
                epsilon=epsilon, delta=delta, sensitivity=sensitivity
            )
        elif mechanism_type == "gaussian":
            self.dp = Gaussian(epsilon, delta=delta, sensitivity=sensitivity, args=args)
        else:
            raise NotImplementedError("DP mechanism not implemented!")


    def add_noise(self, w_global, qw):
        new_params = dict()
        for k in w_global.keys():
            new_params[k] = self._compute_new_params(w_global[k], qw)
        return new_params

    def _compute_new_params(self, param, qw):
        noise = self.dp.compute_noise(param.shape, qw)
        return noise + param

    def _compute_new_grad(self, grad, qw):
        noise = self.dp.compute_noise(grad.shape, qw)

        return noise + grad


    def add_a_noise_to_local_data(self, local_data):
        new_data = []
        for i in range(len(local_data)):
            list = []
            for x in local_data[i]:
                y = self._compute_new_grad(x)
                list.append(y)
            new_data.append(tuple(list))
        return new_data

    def clip_local_update(self, update, clipping_norm, norm_type: float = 2.0):
        total_norm = torch.norm(torch.stack([torch.norm(update[k], norm_type) for k in update.keys()]), norm_type)
        clip_coef = clipping_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for k in update.keys():
            update[k].mul_(clip_coef_clamped)

        return update


