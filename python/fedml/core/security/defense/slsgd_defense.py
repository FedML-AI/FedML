import math
from fedml.core.security.common.utils import get_total_sample_num
from fedml.core.security.defense.defense_base import BaseDefenseMethod

"""
defense with aggregation, added by Shanshan, 07/13/2022
SLSGD: Secure and efficient distributed on-device machine learning
https://arxiv.org/pdf/1903.06996.pdf

This approach has two stage: 1. process model list, and 2. aggregation

In stage 1, users can choose whether to trim the model list (option 2 when b > 0) or not (option 1 or option 2 when b = 0).
In the first case, the steps are as follows: 
    1) compute a score for gradients from clients; 
    2) sort the model list with the scores
    3) (trim) remove the first b gradients and the last b gradients from the model list.
In the second case, the algorithm does nothing to the model list.

In stage 2, a user can set alpha (i.e., the weight of moving average) between 0 and 1 
and does an aggregation with the averaged model and the global model in the last iteration using aggregate(). 
Specifically, alpha = 1 indicates the new global model is set to the new average model, 
and alpha = 0 indicates the global model is identical to the old one.

"""


class SLSGDDefense(BaseDefenseMethod):
    def __init__(self, trim_param_b, alpha, option_type):
        self.b = trim_param_b  # parameter of trimmed mean
        if alpha > 1 or alpha < 0:
            raise ValueError("the bound of alpha is [0, 1]")
        self.alpha = alpha
        self.option_type = option_type

    def defend(self, model_list, global_model=None, refs=None):
        if self.b > math.ceil(len(model_list) / 2) - 1 or self.b < 0:
            raise ValueError(
                "the bound of b is [0, {}])".format(math.ceil(len(model_list) / 2) - 1)
            )
        if self.option_type != 1 and self.option_type != 2:
            raise Exception("Such option type does not exist!")
        if self.option_type == 2:
            model_list = self._sort_and_trim(model_list)  # process model list
        return model_list

    def robust_aggregate(self, client_grad_list, global_w=None):
        (num0, avg_params) = client_grad_list[0]
        total_sample_num = get_total_sample_num(client_grad_list)
        for k in avg_params.keys():
            for i in range(0, len(client_grad_list)):
                local_sample_number, local_model_params = client_grad_list[i]
                w = local_sample_number / total_sample_num
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k] * w
        for k in avg_params.keys():
            avg_params[k] = (1 - self.alpha) * global_w[k] + self.alpha * avg_params[k]
        return avg_params

    def _sort_and_trim(self, model_list):
        model_list2 = []
        for i in range(0, len(model_list)):
            local_sample_number, local_model_params = model_list[i]
            model_list2.append(
                (
                    local_sample_number,
                    local_model_params,
                    self.compute_a_score(local_sample_number, local_model_params),
                )
            )
        model_list2.sort(key=lambda grad: grad[2])  # sort by coordinate-wise scores
        model_list2 = model_list2[self.b : len(model_list) - self.b]
        model_list = [(t[0], t[1]) for t in model_list2]
        return model_list

    @staticmethod
    def compute_a_score(local_sample_number, local_model_params):
        # todo: change to coordinate-wise score
        return local_sample_number
