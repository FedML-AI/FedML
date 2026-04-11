import random
import fedml
import torch
import logging
from collections import OrderedDict
from .attack_base import BaseAttackMethod
from ..common.utils import is_weight_param, vectorize_weight, compute_euclidean_distance
from typing import List, Tuple, Dict, Any

"""
"How To Backdoor Federated Learning? "
http://proceedings.mlr.press/v108/bagdasaryan20a/bagdasaryan20a.pdf 
The attacker scales up the weights of the backdoored model by gamma = total_client_num / participant_num
and replaces the global model after averaging with the other participants’ models.

Optimizations to avoid anomaly detection:
1. Constrain-and-scale: requires to modify the loss function; too much modifications on existing system; not implemented
2. Train-and-scale: the attacker scales up the model weights by gamma up to the bound S permitted by the anomaly detector

Default setting:
randomly select a client as a malicious client each round; attack happens at each round; no scale factor to evade anomaly detection
"""


class ModelReplacementBackdoorAttack(BaseAttackMethod):
    def __init__(self, args):
        if hasattr(args, "malicious_client_id") and isinstance(args.malicious_client_id, int):
            # assume only 1 malicious client
            self.malicious_client_id = args.malicious_client_id
        else:
            self.malicious_client_id = None  # randomly select malicious client
        if hasattr(args, "attack_training_rounds") and isinstance(args.poisoned_training_round, list):
            self.attack_training_rounds = args.attack_training_rounds
        else:
            self.attack_training_rounds = None # attack happens in each round
        # parameters for Train-and-scale to evade anomaly detection
        if hasattr(args, "scale_factor_S") and isinstance(args.scale_factor_S, float):
            self.scale_factor_S = args.scale_factor_S
        else:
            self.scale_factor_S = None
        self.training_round = 1
        self.device = fedml.device.get_device(args)

    def attack_model(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            extra_auxiliary_info: Any = None,
    ):
        participant_num = len(raw_client_grad_list)
        if self.attack_training_rounds is not None and self.training_round not in self.attack_training_rounds:
            return raw_client_grad_list
        if self.malicious_client_id is None:
            malicious_idx = random.randrange(participant_num)  # randomly select a client as a malicious client
        else:
            malicious_idx = self.malicious_client_id
        global_model = OrderedDict()
        for k in extra_auxiliary_info.keys():
            global_model[k] = extra_auxiliary_info[k].to(self.device)
        logging.info(f"malicious_idx={malicious_idx}")
        (num, original_client_model) = raw_client_grad_list[malicious_idx]
        raw_client_grad_list.pop(malicious_idx)
        if self.scale_factor_S is None:
            gamma = participant_num
        else:
            gamma = self.compute_gamma(global_model, original_client_model)
        for k in original_client_model.keys():
            if is_weight_param(k):
                original_client_model[k] = torch.tensor(gamma * (original_client_model[k] - global_model[k]) + global_model[k]).float().to(self.device)
        raw_client_grad_list.insert(malicious_idx, (num, original_client_model))
        self.training_round = self.training_round + 1
        return raw_client_grad_list

    def compute_gamma(self, global_model, original_client_model):
        # total_client_num / η, η: global learning rate;
        # when η = total_client_num/participant_num, the model is fully replaced by the average of the local models
        malicious_client_model_vec = vectorize_weight(original_client_model)
        global_model_vec = vectorize_weight(global_model)
        gamma = self.scale_factor_S / (compute_euclidean_distance(malicious_client_model_vec, global_model_vec))
        return gamma