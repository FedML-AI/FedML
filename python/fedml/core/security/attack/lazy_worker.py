import random
from collections import OrderedDict

import fedml
import numpy as np
from typing import Callable
import torch
from .attack_base import BaseAttackMethod
from ..common.utils import is_weight_param, sample_some_clients
from typing import List, Tuple, Dict, Any


class LazyWorkerAttack(BaseAttackMethod):
    def __init__(self, config):
        self.lazy_worker_num = config.lazy_worker_num
        self.attack_mode = (
            config.attack_mode
        )  # random: randomly generate a weight; zero: set the weight to 0
        self.device = fedml.device.get_device(config)
        if self.attack_mode == "gaussian":
            if (
                hasattr(config, "gaussian_mu")
                and isinstance(config.gaussian_mu, float)
                and hasattr(config, "gaussian_sigma")
                and isinstance(config.gaussian_sigma, float)
            ):
                self.gaussian_mu = config.gaussian_mu
                self.gaussian_sigma = config.gaussian_sigma
            else:
                self.gaussian_mu = 0.0
                self.gaussian_sigma = 1.0
        self.client_cache = []
        self.round = 1
        # global or client, indicating which model of the last round to use
        self.attack_base = config.attack_base

    def attack_model(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        if self.round == 1:
            self.client_cache = [grad for (_, grad) in raw_client_grad_list]
            return raw_client_grad_list
        self.round = self.round + 1

        if self.attack_mode == "uniform":
            mask_func = self.uniform_mask
        elif self.attack_mode == "random":
            mask_func = self.random_mask
        elif self.attack_mode == "gaussian":
            mask_func = self.gaussian_mask
        elif self.attack_mode == "no_mask":
            mask_func = self.no_mask
        else:
            raise NotImplementedError("Method not implemented!")

        if len(raw_client_grad_list) < self.lazy_worker_num:
            self.lazy_worker_num = len(raw_client_grad_list)
        lazy_worker_idxs = sample_some_clients(
            len(raw_client_grad_list), self.lazy_worker_num
        )
        print(f"lazy_worker_idxs={lazy_worker_idxs}")
        new_model_list = []
        for i in range(0, len(raw_client_grad_list)):
            if i not in lazy_worker_idxs:
                new_model_list.append(raw_client_grad_list[i])
                _, params = raw_client_grad_list[i]
                self.client_cache[i] = params
            else:
                local_sample_number, _ = raw_client_grad_list[i]
                if self.attack_base == "global":
                    previous_model_params = extra_auxiliary_info  # global model
                else:  # client
                    previous_model_params = self.client_cache[i]
                previous_model_params = mask_func(previous_model_params)
                new_model_list.append((local_sample_number, previous_model_params))
        return new_model_list

    def _add_a_mask_on_clients(self, model_list, lazy_worker_idxs, mask_func: Callable):
        new_model_list = []
        for i in range(0, len(model_list)):
            if i not in lazy_worker_idxs:
                new_model_list.append(model_list[i])
                _, params = model_list[i]
                self.client_cache[i] = params
            else:
                local_sample_number, _ = model_list[i]
                previous_model_params = self.client_cache[i]
                previous_model_params = mask_func(previous_model_params)
                new_model_list.append((local_sample_number, previous_model_params))
        return new_model_list

    def _add_a_mask_on_global(self, model_list, lazy_worker_idxs, mask_func: Callable):
        new_model_list = []
        for i in range(0, len(model_list)):
            if i not in lazy_worker_idxs:
                new_model_list.append(model_list[i])
            else:
                local_sample_number, _ = model_list[i]
                previous_model_params = self.client_cache[i]
                previous_model_params = mask_func(previous_model_params)
                new_model_list.append((local_sample_number, previous_model_params))
        return new_model_list

    def random_mask(self, previous_model_params):
        # add a random mask in [-1, 1]
        for k in previous_model_params.keys():
            if is_weight_param(k):
                previous_model_params[k] = (
                    torch.from_numpy(
                        2
                        * np.random.random_sample(size=previous_model_params[k].size())
                        - 1
                        + previous_model_params[k]
                    )
                    .float()
                    .to(self.device)
                )
        return previous_model_params

    def gaussian_mask(self, previous_model_params):
        # add a gaussian mask
        for k in previous_model_params.keys():
            if is_weight_param(k):
                previous_model_params[k] = previous_model_params[k] + torch.normal(
                    mean=self.gaussian_mu,
                    std=self.gaussian_sigma,
                    size=previous_model_params[k].size(),
                )
        return previous_model_params

    def uniform_mask(self, previous_model):
        # randomly generate a uniform mask
        unif_param = random.uniform(-1, 1)
        print(f"unif_mode_param = {unif_param}")
        for k in previous_model.keys():
            if is_weight_param(k):
                previous_model[k] = (
                    torch.from_numpy(
                        np.full(previous_model[k].size(), unif_param)
                        + previous_model[k]
                    )
                    .float()
                    .to(self.device)
                )
        return previous_model

    def no_mask(self, previous_model_params):
        # directly return the model in the last round
        return previous_model_params
