from collections import OrderedDict
from .defense_base import BaseDefenseMethod
from ..common import utils
from ...dp.mechanisms import Gaussian

"""
CRFL: Certifiably Robust Federated Learning against Backdoor Attacks (ICML 2021)
http://proceedings.mlr.press/v139/xie21a/xie21a.pdf
"""


from .base_defense_method import BaseDefenseMethod
from .utils import compute_model_norm
from .gaussian import compute_noise_using_sigma
from collections import OrderedDict


class CRFLDefense(BaseDefenseMethod):
    """
    CRFL (Clip and Randomly Flip) Defense for Federated Learning.

    CRFL Defense is a defense method for federated learning that clips the global model's weights if they exceed a
    dynamic threshold and adds Gaussian noise to the clipped weights to improve privacy.

    Args:
        config: Configuration parameters for the defense, including 'clip_threshold' (optional), 'sigma', 'comm_round',
        and 'dataset'.

    Attributes:
        epoch (int): The current training epoch.
        user_defined_clip_threshold (float, optional): A user-defined clipping threshold for model weights.
        sigma (float): The standard deviation of Gaussian noise added to clipped weights.
        total_ite_num (int): The total number of communication rounds.
        dataset_param_function (function): A function to compute the dynamic clipping threshold based on the dataset.
    """

    def __init__(self, config):
        """
        Initialize the CRFLDefense with the specified configuration.

        Args:
            config: Configuration parameters for the defense.
        """
        self.config = config
        self.epoch = 1
        if hasattr(config, "clip_threshold"):
            self.user_defined_clip_threshold = config.clip_threshold
        else:
            self.user_defined_clip_threshold = None
        if hasattr(config, "sigma") and isinstance(config.sigma, float):
            self.sigma = config.sigma
        else:
            self.sigma = 0.01  # Default sigma value as used in CRFL code
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
            raise Exception(
                f"Dataset not supported: {config.dataset} and clip_threshold not defined.")

    def defend_after_aggregation(self, global_model):
        """
        Apply CRFL Defense after model aggregation.

        Args:
            global_model (OrderedDict): The global model to be defended.

        Returns:
            OrderedDict: The defended global model after clipping and adding Gaussian noise.
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
            new_global_model[k] = global_model[k] + \
                compute_noise_using_sigma(self.sigma, global_model[k].shape)
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
        """
        Clip the weight norm of the model.

        Args:
            model (OrderedDict): The model whose weights are to be clipped.
            clip_threshold (float): The threshold value for clipping.

        Returns:
            OrderedDict: The model with clipped weights.
        """
        total_norm = compute_model_norm(model)
        if total_norm > clip_threshold:
            clip_coef = clip_threshold / (total_norm + 1e-6)
            new_model = OrderedDict()
            for k in model.keys():
                new_model[k] = model[k] * clip_coef
            return new_model
        return model
