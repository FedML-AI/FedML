from collections import OrderedDict
from .defense_base import BaseDefenseMethod
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
            self.user_defined_clip_threshold = config.clip_threshold
        else:
            self.user_defined_clip_threshold = None
        if hasattr(config, "sigma") and isinstance(config.sigma, float):
            self.sigma = config.sigma
        else:
            self.sigma = 0.01  # in the code of CRFL, the author set sigma to 0.01
        self.total_ite_num = config.comm_round
        if config.dataset == "mnist":
            self.dataset_param_function = self._crfl_compute_param_for_mnist
        elif config.dataset == "emnist":
            self.dataset_param_function = self._crfl_compute_param_for_emnist
        elif config.dataset == "lending_club_loan":
            self.dataset_param_function = self._crfl_compute_param_for_loan
        elif self.user_defined_clip_threshold is not None:
            self.dataset_param_function = self._crfl_self_defined_dataset_param
        else:
            raise Exception(f"dataset not supported: {config.dataset} and clip_threshold not defined ")

    def defend_after_aggregation(self, global_model):
        """
        clip the global model; dynamic threshold is adjusted according to the dataset;
        in the experiment, the authors set the dynamic threshold as follows:
           dataset == MNIST: dynamic_thres = epoch * 0.1 + 2
           dataseet == LOAN: dynamic_thres = epoch * 0.025 + 2
           datset == EMNIST: dynamic_thres = epoch * 0.25 + 4
        """
        clip_threshold = self.dataset_param_function()
        if self.user_defined_clip_threshold is not None and self.user_defined_clip_threshold < clip_threshold:
            clip_threshold = self.user_defined_clip_threshold

        global_model = self.clip_weight_norm(global_model, clip_threshold)
        if self.epoch == self.total_ite_num:
            return global_model
        self.epoch += 1
        new_global_model = OrderedDict()
        for k in global_model.keys():
            new_global_model[k] = global_model[k] + Gaussian.compute_noise_using_sigma(self.sigma, global_model[k].shape)
        return new_global_model

    def _crfl_self_defined_dataset_param(self):
        return self.user_defined_clip_threshold

    def _crfl_compute_param_for_mnist(self):
        return self.epoch * 0.1 + 2

    def _crfl_compute_param_for_loan(self):
        return self.epoch * 0.025 + 2

    def _crfl_compute_param_for_emnist(self):
        return self.epoch * 0.25 + 4

    @staticmethod
    def clip_weight_norm(model, clip_threshold):
        total_norm = utils.compute_model_norm(model)
        print(f"total_norm = {total_norm}")
        if total_norm > clip_threshold:
            clip_coef = clip_threshold / (total_norm + 1e-6)
            new_model = OrderedDict()
            for k in model.keys():
                new_model[k] = model[k] * clip_coef
            return new_model
        return model
