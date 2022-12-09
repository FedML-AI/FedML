from fedml.core.dp.mechanisms import Gaussian, Laplace
import torch
from typing import Union, Iterable

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
        # if self.enable_accountant:
        #     self.accountant.spend(epsilon=self.epsilon, delta=0)
        return new_params

    def _compute_new_params(self, param, qw):
        noise = self.dp.compute_noise(param.shape, qw)
        return noise + param

    def _compute_new_grad(self, grad, qw):
        noise = self.dp.compute_noise(grad.shape, qw)

        return noise + grad

    # def add_noise(self, grad):
    #     noise_list_len = len(vectorize_weight(grad))
    #     noise_list = np.zeros(noise_list_len)
    #     vec_weight = vectorize_weight(grad)
    #     for i in range(noise_list_len):
    #         noise_list[i] = self.dp.compute_noise()
    #     new_vec_grad = vec_weight + noise_list
    #
    #     new_grad = {}
    #     index_bias = 0
    #     print(f"noises in add_noise = {noise_list}")
    #     for item_index, (k, v) in enumerate(grad.items()):
    #         if is_weight_param(k):
    #             new_grad[k] = new_vec_grad[index_bias : index_bias + v.numel()].view(
    #                 v.size()
    #             )
    #             index_bias += v.numel()
    #         else:
    #             new_grad[k] = v
    #     return new_grad

    def add_a_noise_to_local_data(self, local_data):
        new_data = []
        for i in range(len(local_data)):
            list = []
            for x in local_data[i]:
                y = self._compute_new_grad(x)
                list.append(y)
            new_data.append(tuple(list))
        return new_data

    # def clip_local_gradients(self, client_model_list, clipping_norm):
    #     for model in client_model_list:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipping_norm)
    def clip_local_update(self, update, clipping_norm, norm_type: float = 2.0):
        total_norm = torch.norm(torch.stack([torch.norm(update[k], norm_type) for k in update.keys()]), norm_type)
        clip_coef = clipping_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for k in update.keys():
            update[k].mul_(clip_coef_clamped)

        return update

# def clip_local_grads(self, local_params):
#     """
#     Performs gradient clipping.
#     Stores clipped and aggregated gradients into `p.summed_grad```
#     """
#     per_param_norms = [
#         g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
#     ]
#     per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
#     per_sample_clip_factor = (
#             self.max_grad_norm / (per_sample_norms + 1e-6)
#     ).clamp(max=1.0)
#
#     for p in self.params:
#         grad_sample = self._get_flat_grad_sample(p)
#         grad = contract("i,i...", per_sample_clip_factor, grad_sample)
#         # import pdb
#         # pdb.set_trace()
#
#         if p.summed_grad is not None:
#             p.summed_grad += grad
#         else:
#             p.summed_grad = grad
