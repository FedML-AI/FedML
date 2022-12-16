import functools
import random
from collections import OrderedDict
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from .attack_base import BaseAttackMethod

"""
ref: Baruch, Gilad, Moran Baruch, and Yoav Goldberg. 
"A little is enough: Circumventing defenses for distributed learning." Advances in Neural Information Processing Systems 32 (2019).

overview: the attacker can find the set of parameters within the same range (via maximizing standard deviations /sigma of params) that will
          introduce a backdoor to the system with only a minimal impact on accuracy for the original task. 

Steps:
(1) Calculate mean and standard deviations of each dimension of parameters of corrupted workers.
(2) Each malicious worker trains the model with the backdoor. -> it is not implemeted in this code but should be trained with FedML
    Loss = alpha * L_backdoor + (1 - alpha) * l, where l = sum of [(NewParam - OldParam)/max(z * sigma, 1e-5)]^2
(3) Update malicious parameters to the range {mean +/- z^(max) * std}, where z is the lower and upper bounds for applicable changes around the mean

params:
backdoor_client_num -> number of attackers
alpha -> distillation weight on loss trained on the backdoors -> L = alpha * l_backdoor + (1 - alpha) * regularization loss
num_std -> how many standard deviations should the attacker change
"""


class BackdoorAttack(BaseAttackMethod):
    def __init__(
        self, backdoor_client_num, client_num, num_std=None, dataset=None, backdoor_type="pattern",
    ):
        self.backdoor_client_num = backdoor_client_num
        self.client_num = client_num
        self.num_std = num_std
        self.backdoor = backdoor_type  #
        # build backdoor, disable, which should be embedded into FedML training.
        if dataset is not None:
            if backdoor_type == "pattern":
                target = dataset[1]
                target *= 0  # make images with the pattern always output 0
            else:
                target = dataset[1]
                target = (target + 1) % 5
            self.train_loader = DataLoader(
                dataset=TensorDataset(dataset[0], target), batch_size=3, shuffle=True, num_workers=2,
            )
            self.test_loader = self.train_loader
        else:
            pass

    def attack_model(self, raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None):
        # the local_w comes from local training (regular)
        backdoor_idxs = self._get_malicious_client_idx(len(raw_client_grad_list))
        (num0, averaged_params) = raw_client_grad_list[0]

        # fake grad
        # refs/or other variable here should be gradient that makes the training agrees on the correct gradient
        # (in maliciuos clients), which limits the change that can be applied.
        grads = []
        for i in backdoor_idxs:
            (_, param) = raw_client_grad_list[i]
            # grad = np.concatenate([param.grad.data.cpu().numpy().flatten() for param in model.parameters()]) // for real net
            grad = np.concatenate([param[p_name].numpy().flatten() * 0.5 for p_name in param])
            grads.append(grad)
        grads_mean = np.mean(grads, axis=0)
        grads_stdev = np.var(grads, axis=0) ** 0.5

        learning_rate = 0.1
        original_params_flat = np.concatenate([averaged_params[p_name].numpy().flatten() for p_name in averaged_params])
        initial_params_flat = (
            original_params_flat - learning_rate * grads_mean
        )  # the corrected param after the user optimized, because we still want the model to improve
        mal_net_params = self.train_malicious_network(initial_params_flat, original_params_flat)

        # Getting from the final required mal_net_params to the gradients that needs to be applied on the parameters of the previous round.
        new_params = mal_net_params + learning_rate * grads_mean
        new_grads = (initial_params_flat - new_params) / learning_rate
        # authors in the paper claims to limit the range of parameters but the code limits the gradient.
        new_user_grads = np.clip(
            new_grads, grads_mean - self.num_std * grads_stdev, grads_mean + self.num_std * grads_stdev,
        )
        # the returned gradient controls the local update for malicious clients
        return new_user_grads

    @staticmethod
    def add_pattern(img):
        # disable
        img[:, :5, :5] = 2.8
        return img

    def train_malicious_network(self, initial_params_flat, param):
        # skip training process
        # return flatten_params(param)
        return param

    def _get_malicious_client_idx(self, client_num):
        return random.sample(range(client_num), self.backdoor_client_num)


def flatten_params(params):
    # for real net
    return np.concatenate([i.data.cpu().numpy().flatten() for i in params])


def row_into_parameters(row, parameters):
    # for real net
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x * y, param.shape)
        current_data = row[offset : offset + new_size]

        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size
