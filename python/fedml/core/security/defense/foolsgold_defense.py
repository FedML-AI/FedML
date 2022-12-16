from collections import OrderedDict
from typing import Callable, List, Tuple, Dict, Any
import numpy as np
from .defense_base import BaseDefenseMethod

"""
The Limitations of Federated Learning in Sybil Settings.
https://www.usenix.org/system/files/raid20-fung.pdf 
https://github.com/DistributedML/FoolsGold
potential bugs when using memory: when only some of clients participate in computing, grads in different iterations may be from different clients
"""


class FoolsGoldDefense(BaseDefenseMethod):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.memory = None

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        client_num = len(raw_client_grad_list)
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        # print(len(importance_feature_list))

        if self.memory is None:
            self.memory = importance_feature_list
        else:  # memory: potential bugs: grads in different iterations may be from different clients
            for i in range(client_num):
                self.memory[i] += importance_feature_list[i]
        alphas = self.fools_gold_score(self.memory)  # Use FG

        print("alphas = {}".format(alphas))
        assert len(alphas) == len(
            raw_client_grad_list
        ), "len of wv {} is not consistent with len of client_grads {}".format(len(alphas), len(raw_client_grad_list))
        new_grad_list = []
        client_num = len(raw_client_grad_list)
        for i in range(client_num):
            sample_num, grad = raw_client_grad_list[i]
            new_grad_list.append((sample_num * alphas[i] / client_num, grad))
        return new_grad_list

    # Takes in grad, compute similarity, get weightings
    @classmethod
    def fools_gold_score(cls, feature_vec_list):
        import sklearn.metrics.pairwise as smp
        n_clients = len(feature_vec_list)
        cs = smp.cosine_similarity(feature_vec_list) - np.eye(n_clients)
        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        alpha = 1 - (np.max(cs, axis=1))
        alpha[alpha > 1.0] = 1.0
        alpha[alpha <= 0.0] = 1e-15

        # Rescale so that max value is alpha
        # print(np.max(alpha))
        alpha = alpha / np.max(alpha)
        alpha[(alpha == 1.0)] = 0.999999

        # Logit function
        alpha = np.log(alpha / (1 - alpha)) + 0.5
        alpha[(np.isinf(alpha) + alpha > 1)] = 1
        alpha[(alpha < 0)] = 0

        return alpha

    def _get_importance_feature(self, raw_client_grad_list):
        # Foolsgold uses the last layer's gradient/weights as the importance feature.
        ret_feature_vector_list = []
        for idx in range(len(raw_client_grad_list)):
            raw_grad = raw_client_grad_list[idx]
            (p, grads) = raw_grad

            # Get last key-value tuple
            (weight_name, importance_feature) = list(grads.items())[-2]
            # print(importance_feature)
            feature_len = np.array(importance_feature.cpu().data.detach().numpy().shape).prod()
            feature_vector = np.reshape(importance_feature.cpu().data.detach().numpy(), feature_len)
            ret_feature_vector_list.append(feature_vector)
        return ret_feature_vector_list
