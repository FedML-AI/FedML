import random
import numpy as np
import torch
from .attack_base import BaseAttackMethod
from ..common.utils import is_weight_param, get_total_sample_num

"""
attack @ server, added by Shanshan, 07/04/2022
"""


class ByzantineAttack(BaseAttackMethod):
    def __init__(self, args):
        self.byzantine_client_num = args.byzantine_client_num
        self.attack_mode = args.attack_mode  # random: randomly generate a weight; zero: set the weight to 0

    def attack_model(self, model_list, global_w, refs=None):
        if len(model_list) < self.byzantine_client_num:
            self.byzantine_client_num = len(model_list)
        if self.attack_mode == "zero":
            byzantine_local_w = self._attack_zero_mode(model_list)
        elif self.attack_mode == "random":
            byzantine_local_w = self._attack_random_mode(model_list)
        elif self.attack_mode == "flip":
            byzantine_local_w = self._attack_flip_mode(model_list, global_w)
        else:
            raise NotImplementedError("Method not implemented!")
        return byzantine_local_w

    def _attack_zero_mode(self, model_list):
        byzantine_idxs = self._get_malicious_client_idx(len(model_list))
        new_model_list = []

        for i in range(0, len(model_list)):
            if i not in byzantine_idxs:
                new_model_list.append(model_list[i])
            else:
                local_sample_number, local_model_params = model_list[i]
                for k in local_model_params.keys():
                    if is_weight_param(k):
                        local_model_params[k] = torch.from_numpy(np.zeros(local_model_params[k].size())).float()
                new_model_list.append((local_sample_number, local_model_params))
        return new_model_list

    def _attack_random_mode(self, model_list):
        byzantine_idxs = self._get_malicious_client_idx(len(model_list))
        new_model_list = []

        for i in range(0, len(model_list)):
            if i not in byzantine_idxs:
                new_model_list.append(model_list[i])
            else:
                local_sample_number, local_model_params = model_list[i]
                for k in local_model_params.keys():
                    if is_weight_param(k):
                        local_model_params[k] = torch.from_numpy(np.random.random_sample(size=local_model_params[k].size())).float()
                new_model_list.append((local_sample_number, local_model_params))
        return new_model_list


    def _attack_flip_mode(self, model_list, global_model):
        byzantine_idxs = self._get_malicious_client_idx(len(model_list))
        new_model_list = []
        for i in range(0, len(model_list)):
            if i not in byzantine_idxs:
                new_model_list.append(model_list[i])
            else:
                local_sample_number, local_model_params = model_list[i]
                for k in local_model_params.keys():
                    if is_weight_param(k):
                        local_model_params[k] = global_model[k] + (global_model[k] - local_model_params[k])
                new_model_list.append((local_sample_number, local_model_params))
        return new_model_list

    def _get_malicious_client_idx(self, client_num):
        return random.sample(range(client_num), self.byzantine_client_num)
