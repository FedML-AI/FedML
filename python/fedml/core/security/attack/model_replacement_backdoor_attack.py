import random
import fedml
import torch
import logging
from collections import OrderedDict
from .attack_base import BaseAttackMethod
from ..common.utils import is_weight_param, vectorize_weight, compute_euclidean_distance, should_scale_param
from typing import List, Tuple, Dict, Any

"""
"How To Backdoor Federated Learning?"
http://proceedings.mlr.press/v108/bagdasaryan20a/bagdasaryan20a.pdf
The attacker scales up the weights of the backdoored model by gamma = total_client_num / participant_num
and replaces the global model after averaging with the other participants' models.

Optimizations to avoid anomaly detection:
1. Constrain-and-scale: requires to modify the loss function; too much modifications on existing system; not implemented
2. Train-and-scale: the attacker scales up the model weights by gamma up to the bound S permitted by the anomaly detector

Default setting:
randomly select a client as a malicious client each round; attack happens at each round; no scale factor to evade anomaly detection
"""


class ModelReplacementBackdoorAttack(BaseAttackMethod):
    def __init__(self, args):
        # --- attack_training_rounds (0-indexed) ---
        if hasattr(args, "attack_training_rounds") and isinstance(args.attack_training_rounds, list):
            self.attack_training_rounds = args.attack_training_rounds
        else:
            self.attack_training_rounds = None  # attack happens in each round

        # --- scale_gamma ---
        if hasattr(args, "scale_gamma"):
            raw = args.scale_gamma
            if isinstance(raw, str) and raw.lower() == "auto":
                self.scale_gamma = "auto"
            else:
                self.scale_gamma = float(raw)
        else:
            self.scale_gamma = float(getattr(args, "client_num_in_total", 10))

        # --- malicious client IDs: fixed to [0, 1, ..., K-1] ---
        byzantine_client_num = int(getattr(args, "byzantine_client_num", 3))
        self.malicious_client_ids = list(range(byzantine_client_num))

        # --- assert full participation ---
        client_num_per_round = int(getattr(args, "client_num_per_round", 10))
        client_num_in_total = int(getattr(args, "client_num_in_total", 10))
        assert client_num_per_round == client_num_in_total, (
            "Scaling attack requires full participation: "
            "client_num_per_round (%d) != client_num_in_total (%d)"
            % (client_num_per_round, client_num_in_total)
        )

        # parameters for Train-and-scale to evade anomaly detection (legacy)
        if hasattr(args, "scale_factor_S") and isinstance(getattr(args, "scale_factor_S", None), float):
            self.scale_factor_S = args.scale_factor_S
        else:
            self.scale_factor_S = None

        self.training_round = 0  # 0-indexed
        self.device = fedml.device.get_device(args)

        logging.info(
            "Scaling attack init | malicious_client_ids=%s | gamma=%s | attack_rounds=%s",
            self.malicious_client_ids,
            self.scale_gamma,
            self.attack_training_rounds,
        )

    def attack_model(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            extra_auxiliary_info: Any = None,
    ):
        # Non-attack round: pass through
        if self.attack_training_rounds is not None and self.training_round not in self.attack_training_rounds:
            self.training_round += 1
            return raw_client_grad_list

        # Build global model on device
        global_model = OrderedDict()
        for k in extra_auxiliary_info.keys():
            global_model[k] = extra_auxiliary_info[k].to(self.device)

        if self.scale_gamma == "auto":
            total_samples = sum(num for num, _ in raw_client_grad_list)
            malicious_samples = sum(
                raw_client_grad_list[idx][0]
                for idx in self.malicious_client_ids
                if idx < len(raw_client_grad_list)
            )
            gamma = total_samples / malicious_samples
            logging.info(
                "Scaling auto-gamma | round=%d | total_samples=%d | malicious_samples=%d | gamma=%.4f",
                self.training_round, int(total_samples), int(malicious_samples), gamma,
            )
        else:
            gamma = self.scale_gamma

        # Expose last computed gamma for structured logging (D-13).
        self.last_gamma = float(gamma)

        # Scale each malicious client's model in-place (no pop+insert)
        for idx in self.malicious_client_ids:
            if idx >= len(raw_client_grad_list):
                continue
            (num, client_model) = raw_client_grad_list[idx]
            for k in client_model.keys():
                if should_scale_param(k):
                    client_model[k] = (
                        gamma * (client_model[k].to(self.device) - global_model[k])
                        + global_model[k]
                    ).float()
            raw_client_grad_list[idx] = (num, client_model)
            logging.info(
                "Scaling apply | round=%d | malicious_idx=%d | gamma=%s",
                self.training_round,
                idx,
                gamma,
            )

        self.training_round += 1
        return raw_client_grad_list

    def compute_gamma(self, global_model, original_client_model):
        malicious_client_model_vec = vectorize_weight(original_client_model)
        global_model_vec = vectorize_weight(global_model)
        gamma = self.scale_factor_S / (compute_euclidean_distance(malicious_client_model_vec, global_model_vec))
        return gamma
