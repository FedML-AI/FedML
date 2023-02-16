from collections import OrderedDict
import torch
from fedml.core.dp.frames.base_dp_solution import BaseDPFrame
from typing import List, Tuple, Dict, Any

"""
(ICLR 2018) Learning Differentially Private Recurrent Language Models

1. (todo: Need to change sampling process) each user is selected independently with probability q, rather than always selecting a fixed number of users
2. enforce clipping of per-user updates so the total update has bounded L2 norm.
3. use different estimators for the average update (introduced next).
4. add Gaussian noise to the final average update.
"""

class DP_Clip(BaseDPFrame):
    def __init__(self, args):
        super().__init__(args)
        self.clipping_norm = args.clipping_norm
        self.train_data_num_in_total = args.train_data_num_in_total
        self._scale = args.clipping_norm * args.noise_multiplier

    def clip_local_update(self, local_grad, norm_type: float = 2.0):
        total_norm = torch.norm(torch.stack([torch.norm(local_grad[k], norm_type) for k in local_grad.keys()]), norm_type)
        clip_coef = self.clipping_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for k in local_grad.keys():
            local_grad[k].mul_(clip_coef_clamped)
        return local_grad

    def add_local_noise(self, local_grad: OrderedDict, extra_auxiliary_info: Any = None,):
        global_model_params = extra_auxiliary_info
        for k in global_model_params.keys():
            local_grad[k] = local_grad[k] - global_model_params[k]
        return self.clip_local_update(local_grad, self.clipping_norm)

    def add_global_noise(self, global_model: OrderedDict):
        qw = self.train_data_num_in_total * (self.args.client_num_per_round / self.args.client_num_in_total)
        for k in global_model.keys():
            global_model[k] = global_model[k] / qw
        w_global = self.add_noise(
            global_model, qw
        )
        for k in w_global.keys():
            w_global[k] = w_global[k] + global_model[k]

    def get_global_params(self):
        pass

    def compute_noise(self, size, qw):
        self._scale = self._scale / qw
        return torch.normal(mean=0, std=self._scale, size=size)

    def add_noise(self, w_global, qw):
        new_params = OrderedDict()
        for k in w_global.keys():
            new_params[k] = self.compute_noise(w_global[k].shape, qw) + w_global[k]
        return new_params




