import numpy as np
from .defense_base import BaseDefenseMethod
from typing import Callable, List, Tuple, Dict, Any
from ..common import utils
from scipy import spatial


"""
The Limitations of Federated Learning in Sybil Settings.
https://www.usenix.org/system/files/raid20-fung.pdf 
https://github.com/DistributedML/FoolsGold
potential bugs when using memory: when only some of clients participate in computing, grads in different iterations may be from different clients
"""


class FoolsGoldDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.config = config
        self.memory = None
        self.use_memory = config.use_memory

    def run(
        self,
        raw_client_grad_list: List[Tuple[float, Dict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ):
        new_grad_list = self.defend_before_aggregation(
            raw_client_grad_list, extra_auxiliary_info
        )
        return base_aggregation_func(self.config, new_grad_list)

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, Dict]],
        extra_auxiliary_info: Any = None,
    ):
        client_num = len(raw_client_grad_list)
        if self.use_memory:
            if self.memory is None:
                self.memory = [grad for num, grad in raw_client_grad_list]
            else:  # memory: potential bugs: grads in different iterations may be from different clients
                for i in range(client_num):
                    (num, grad) = raw_client_grad_list[i]
                    for k in grad.keys():
                        self.memory[i][k] += grad[k]
            alphas = self.fools_gold_score(self.memory)  # Use FG
        else:
            grads = [grad for (_, grad) in raw_client_grad_list]
            alphas = self.fools_gold_score(grads)  # Use FG

        assert len(alphas) == len(
            raw_client_grad_list
        ), "len of wv {} is not consistent with len of client_grads {}".format(
            len(alphas), len(raw_client_grad_list)
        )
        new_grad_list = []
        client_num = len(raw_client_grad_list)
        for i in range(client_num):
            sample_num, grad = raw_client_grad_list[i]
            new_grad_list.append((sample_num * alphas[i] / client_num, grad))
        return new_grad_list

    # Takes in grad, compute similarity, get weightings
    @staticmethod
    def fools_gold_score(grad_list):
        n_clients = len(grad_list)
        grads = [utils.vectorize_weight(grad) for grad in grad_list]
        cs = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(n_clients):
                cs[i][j] = 1 - spatial.distance.cosine(
                    grads[i].tolist(), grads[j].tolist()
                )
        cs -= np.eye(n_clients)
        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i != j and maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        alpha = 1 - (np.max(cs, axis=1))
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0

        # Rescale so that max value is alpha
        alpha = alpha / np.max(alpha)
        alpha[(alpha == 1)] = 0.99

        # Logit function
        for i in range(len(alpha)):
            if alpha[i] != 0:
                alpha[i] = np.log(alpha[i] / (1 - alpha[i])) + 0.5
        alpha[(np.isinf(alpha) + alpha > 1)] = 1
        alpha[(alpha < 0)] = 0

        return alpha
