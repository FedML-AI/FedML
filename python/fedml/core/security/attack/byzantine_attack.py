import random
import numpy as np
import torch
from fedml.core.security.attack.attack_base import BaseAttackMethod
from fedml.core.security.common.utils import is_weight_param, get_total_sample_num

"""
attack @ server, added by Shanshan, 07/04/2022
"""


class ByzantineAttack(BaseAttackMethod):
    def __init__(self, byzantine_client_num, attack_mode):
        self.byzantine_client_num = byzantine_client_num
        self.attack_mode = attack_mode  # random: randomly generate a weight; zero: set the weight to 0

    def attack(self, local_w, global_w, refs=None):
        if self.attack_mode == "zero":
            byzantine_local_w = self._attack_zero_mode(local_w)
        elif self.attack_mode == "random":
            byzantine_local_w = self._attack_random_mode(local_w)
        else:
            raise NotImplementedError("Method not implemented!")
        return byzantine_local_w

    def _attack_zero_mode(self, model_list):
        byzantine_idxs = self._get_malicious_client_idx(len(model_list))
        (num0, averaged_params) = model_list[0]
        total_sample_num = get_total_sample_num(model_list)
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / total_sample_num
                if i in byzantine_idxs:
                    if is_weight_param(k):
                        local_model_params[k] = torch.from_numpy(
                            np.zeros(local_model_params[k].size())
                        ).float()
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _attack_random_mode(self, model_list):
        byzantine_idxs = self._get_malicious_client_idx(len(model_list))
        (num0, averaged_params) = model_list[0]
        total_sample_num = get_total_sample_num(model_list)
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / total_sample_num
                if i in byzantine_idxs:
                    if is_weight_param(k):
                        local_model_params[k] = torch.from_numpy(
                            np.random.random_sample(size=local_model_params[k].size())
                        ).float()
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _get_malicious_client_idx(self, client_num):
        return random.sample(range(client_num), self.byzantine_client_num)

    @staticmethod
    def _test_fedavg(model_list):
        training_num = get_total_sample_num(model_list)
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                # print(f"i={i}, model_list = {model_list}")
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    @staticmethod
    def _test_fedavg2(model_list):
        training_num = get_total_sample_num(model_list)
        (num0, averaged_params) = model_list[0]
        res = dict()
        for k in averaged_params.keys():
            res[k] = torch.from_numpy(
                np.zeros(averaged_params[k].size())
            ).float()
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                print(f"i={i}, model_list = {model_list}")
                w = local_sample_number / training_num
                res[k] += local_model_params[k] * w
        return res
