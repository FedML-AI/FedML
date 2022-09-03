from .defense_base import BaseDefenseMethod
from typing import Callable, List, Tuple, Dict, Any
from ..common import utils
from ...dp.mechanisms import Gaussian

"""
CRFL: Certifiably Robust Federated Learning against Backdoor Attacks (ICML 2021)
http://proceedings.mlr.press/v139/xie21a/xie21a.pdf
"""


class CRFLDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.config = config
        self.epoch = 1
        if hasattr(config, "clip_threshold"):
            self.clip_threshold = config.clip_threshold
        else:
            self.clip_threshold = None
        if hasattr(config, "sigma") and isinstance(config.sigma, float):
            self.sigma = config.sigma
        else:
            self.sigma = 0.01  # in the code of CRFL, the author set sigma to 0.01

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
        return raw_client_grad_list

    def defend_on_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, Dict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ):
        avg_params = base_aggregation_func(args=self.config, raw_grad_list=raw_client_grad_list)
        """
        clip the global model; dynamic threshold is adjusted according to the dataset;
        in the experiment, the authors set the dynamic threshold as follows:
            dataset == MNIST: dynamic_thres = epoch * 0.1 + 2
            dataseet == LOAN: dynamic_thres = epoch * 0.025 + 2
            datset == EMNIST: dynamic_thres = epoch * 0.25 + 4
        """
        dynamic_threshold = self.epoch * 0.1 + 2
        if self.clip_threshold is not None and self.clip_threshold > dynamic_threshold:
            self.clip_threshold = dynamic_threshold
        self.epoch += 1

        new_model = self.clip_weight_norm(avg_params, self.clip_threshold)
        # the output model is new model; later the algo adds dp noise to the global model
        return new_model

    def defend_after_aggregation(self, global_model):
        # todo: to discuss with chaoyang: the output is the clipped model (real model);
        # add dp noise to the real model and sent the permuted model to clients; how to get the last iteration?
        new_global_model = dict()
        for k in global_model.keys():
            new_global_model[k] = global_model[k] + Gaussian.add_noise_using_sigma(self.sigma, global_model[k].shape)
        return new_global_model

    @staticmethod
    def clip_weight_norm(model, clip):
        total_norm = utils.compute_model_norm(model)
        max_norm = clip
        clip_coef = max_norm / (total_norm + 1e-6)
        new_model = dict()
        if total_norm > max_norm:
            for k in model.keys():
                new_model[k] = model[k] * clip_coef
        return new_model