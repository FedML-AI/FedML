from fedml.core.dp.mechanisms import Gaussian, Laplace
import torch
from typing import Union, Iterable
from collections import OrderedDict

"""call dp mechanisms, e.g., Gaussian, Laplace """

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


class DPMechanism:
    def __init__(self, mechanism_type, epsilon, delta, sensitivity=1):
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
        new_grad = OrderedDict()
        for k in grad.keys():
            new_grad[k] = self._compute_new_grad(grad[k])
        return new_grad

    def _compute_new_grad(self, grad):
        noise = self.dp.compute_noise(grad.shape)
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

    def get_rdp_scale(self):
        return self.dp.get_rdp_scale()




