import logging

import numpy as np

from .common.utils import is_weight_param, vectorize_weight
from .mechanisms import Laplace, Gaussian


class FedMLDifferentialPrivacy:
    _dp_instance = None

    @staticmethod
    def get_instance():
        if FedMLDifferentialPrivacy._dp_instance is None:
            FedMLDifferentialPrivacy._dp_instance = FedMLDifferentialPrivacy()
        return FedMLDifferentialPrivacy._dp_instance

    def __init__(self):
        self.is_dp_enabled = False
        self.dp_type = None
        self.dp = None

    def init(self, args):
        if hasattr(args, "enable_dp") and args.enable_dp:
            logging.info(
                ".......init dp......." + args.mechanism_type + "-" + args.dp_type
            )
            self.is_dp_enabled = True
            mechanism_type = args.mechanism_type.lower()
            self.dp_type = args.dp_type.lower().strip()
            if self.dp_type not in ["cdp", "ldp"]:
                raise ValueError(
                    "DP type can only be cdp (for central DP) and ldp (for local DP)! "
                )
            if mechanism_type == "laplace":
                self.dp = Laplace(
                    epsilon=args.epsilon, delta=args.delta, sensitivity=args.sensitivity
                )
            elif mechanism_type == "gaussian":
                self.dp = Gaussian(
                    epsilon=args.epsilon, delta=args.delta, sensitivity=args.sensitivity
                )
            else:
                raise NotImplementedError("DP mechanism not implemented!")

    def is_enabled(self):
        return self.is_dp_enabled

    def is_cdp_enabled(self):
        return self.is_enabled() and self.get_dp_type() == "cdp"

    def is_ldp_enabled(self):
        return self.is_enabled() and self.get_dp_type() == "ldp"

    def get_dp_type(self):
        return self.dp_type

    def add_noise(self, grad):
        noise_list_len = len(vectorize_weight(grad))
        noise_list = np.zeros(noise_list_len)
        vec_weight = vectorize_weight(grad)
        for i in range(noise_list_len):
            noise_list[i] = self.dp.compute_noise()
        new_vec_grad = vec_weight + noise_list

        new_grad = {}
        index_bias = 0
        print(f"noises in add_noise = {noise_list}")
        for item_index, (k, v) in enumerate(grad.items()):
            if is_weight_param(k):
                new_grad[k] = new_vec_grad[index_bias : index_bias + v.numel()].view(
                    v.size()
                )
                index_bias += v.numel()
            else:
                new_grad[k] = v
        return new_grad

    def add_a_noise_to_local_data(self, local_data):
        new_data = []
        for i in range(len(local_data)):
            list = []
            for x in local_data[i]:
                y = self._compute_new_grad(x)
                list.append(y)
            new_data.append(tuple(list))
        return new_data

    def add_noise_with_data_distribution(self, grad):
        new_grad = dict()
        for k in grad.keys():
            new_grad[k] = self._compute_new_grad(grad[k])
        return new_grad

    def _compute_new_grad(self, grad):
        noise = self.dp.compute_noise_with_shape(grad.shape)
        print(f"noise computed with data distribution = {noise}")
        return noise + grad
