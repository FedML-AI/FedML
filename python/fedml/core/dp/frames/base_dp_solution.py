from abc import ABC
from collections import OrderedDict
import torch
from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism
from typing import List, Tuple


class BaseDPFrame(ABC):
    def __init__(self, args=None):
        self.cdp = None
        self.ldp = None
        self.args = args
        self.is_rdp_accountant_enabled = False
        if hasattr(args, "max_grad_norm") and args.max_grad_norm is not None:
            self.max_grad_norm = args.max_grad_norm
        else:
            self.max_grad_norm = None

    def set_cdp(self, dp_mechanism: DPMechanism):
        self.cdp = dp_mechanism

    def set_ldp(self, dp_mechanism: DPMechanism):
        self.ldp = dp_mechanism

    def add_local_noise(self, local_grad: OrderedDict):
        return self.ldp.add_noise(grad=local_grad)

    def add_global_noise(self, global_model: OrderedDict):
        return self.cdp.add_noise(grad=global_model)

    def set_params_for_dp(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        pass

    def get_rdp_accountant_val(self):
        if self.cdp is not None:
            dp_param = self.cdp.get_rdp_scale()
        elif self.ldp is not None:
            dp_param = self.ldp.get_rdp_scale()
        else:
            raise Exception("can not create rdp accountant")
        return dp_param

    def global_clip(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        if self.max_grad_norm is None:
            return raw_client_model_or_grad_list
        new_grad_list = []
        for (num, local_grad) in raw_client_model_or_grad_list:
            for k in local_grad.keys():
                total_norm = torch.norm(torch.stack([torch.norm(local_grad[k], 2.0) for k in local_grad.keys()]),
                                        2.0)
                clip_coef = self.max_grad_norm / (total_norm + 1e-6)
                clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
                for k in local_grad.keys():
                    local_grad[k].mul_(clip_coef_clamped)
            new_grad_list.append((num, local_grad))
        return new_grad_list



