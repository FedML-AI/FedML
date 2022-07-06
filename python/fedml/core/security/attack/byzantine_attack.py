import random

import numpy as np
import torch

from fedml.core.security.attack.attack_base import BaseAttackMethod
from fedml.core.security.common.utils import is_weight_param, get_total_sample_num

"""
attack @ server, added by Shanshan, 07/04/2022
"""


def get_malicious_client_idx(client_num, malicious_client_num):
    byzantine_idx = random.sample(range(client_num), malicious_client_num)
    return byzantine_idx


class ByzantineAttack(BaseAttackMethod):
    def __init__(self, byzantine_client_num, attack_mode):
        self.byzantine_client_num = byzantine_client_num
        self.attack_mode = attack_mode  # random: randomly generate a weight; zero: set the weight to 0

    def attack(self, local_w, global_w, refs=None):
        byzantine_idxs = get_malicious_client_idx(len(local_w), self.byzantine_client_num)
        if self.attack_mode == "zero":
            byzantine_local_w = self._attack_zero_mode(local_w, byzantine_idxs)
        else:
            raise NotImplementedError("Method not implemented!")
        return byzantine_local_w

    @staticmethod
    def _attack_zero_mode(model_list, byzantine_idxs):
        (num0, averaged_params) = model_list[0]
        total_sample_num = get_total_sample_num(model_list)
        for k in averaged_params.keys():
            training_num = 0
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                print(f"i={i}, local_model_params = {local_model_params}")
                w = local_sample_number/total_sample_num
                training_num += local_sample_number
                if i in byzantine_idxs:
                    if is_weight_param(k):
                        local_model_params[k] = torch.from_numpy(
                            np.zeros(local_model_params[k].size())
                        ).float()
                        print(f"is_weight_param--local_model_params[k] = {local_model_params[k]}")
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
            # averaged_params[k] /= training_num
        return averaged_params

    @staticmethod
    def _attack_test(model_list):
        training_num = get_total_sample_num(model_list)
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                print(f"i={i}, model_list = {model_list}")
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params


