from collections import OrderedDict

import fedml
import numpy as np
import torch
from .attack_base import BaseAttackMethod
from ..common.utils import is_weight_param, sample_some_clients
from typing import List, Tuple, Any

"""
attack @ server, added by Shanshan, 07/04/2022
"""


class ByzantineAttack(BaseAttackMethod):
    def __init__(self, args):
        self.byzantine_client_num = args.byzantine_client_num
        self.attack_mode = args.attack_mode  # random: randomly generate a weight; zero: set the weight to 0
        self.device = fedml.device.get_device(args)

    def attack_model(self, raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None):
        if len(raw_client_grad_list) < self.byzantine_client_num:
            self.byzantine_client_num = len(raw_client_grad_list)
        byzantine_idxs = sample_some_clients(len(raw_client_grad_list), self.byzantine_client_num)
        print(f"byzantine_idxs={byzantine_idxs}")
        if self.attack_mode == "zero":
            byzantine_local_w = self._attack_zero_mode(raw_client_grad_list, byzantine_idxs)
        elif self.attack_mode == "random":
            byzantine_local_w = self._attack_random_mode(raw_client_grad_list, byzantine_idxs)
        elif self.attack_mode == "flip":
            byzantine_local_w = self._attack_flip_mode(raw_client_grad_list, byzantine_idxs, extra_auxiliary_info) # extra_auxiliary_info: global model
        else:
            raise NotImplementedError("Method not implemented!")
        return byzantine_local_w

    def _attack_zero_mode(self, model_list, byzantine_idxs):
        new_model_list = []
        for i in range(0, len(model_list)):
            if i not in byzantine_idxs:
                new_model_list.append(model_list[i])
            else:
                local_sample_number, local_model_params = model_list[i]
                for k in local_model_params.keys():
                    if is_weight_param(k):
                        local_model_params[k] = torch.from_numpy(np.zeros(local_model_params[k].size())).float().to(self.device)
                new_model_list.append((local_sample_number, local_model_params))
        return new_model_list

    def _attack_random_mode(self, model_list, byzantine_idxs):
        new_model_list = []

        for i in range(0, len(model_list)):
            if i not in byzantine_idxs:
                new_model_list.append(model_list[i])
            else:
                local_sample_number, local_model_params = model_list[i]
                for k in local_model_params.keys():
                    if is_weight_param(k):
                        local_model_params[k] = torch.from_numpy(2*np.random.random_sample(size=local_model_params[k].size())-1).float().to(self.device)
                new_model_list.append((local_sample_number, local_model_params))
        return new_model_list


    def _attack_flip_mode(self, model_list, byzantine_idxs, global_model):
        new_model_list = []
        for i in range(0, len(model_list)):
            if i not in byzantine_idxs:
                new_model_list.append(model_list[i])
            else:
                local_sample_number, local_model_params = model_list[i]
                for k in local_model_params.keys():
                    if is_weight_param(k):
                        local_model_params[k] = global_model[k].float().to(self.device) + (global_model[k].float().to(self.device) - local_model_params[k].float().to(self.device))
                new_model_list.append((local_sample_number, local_model_params))
        return new_model_list
