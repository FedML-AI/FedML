from collections import OrderedDict
import torch
from fedml.core.dp.frames.base_dp_solution import BaseDPFrame
from typing import List, Tuple, Dict, Any

"""
(ICLR 2018) Learning Differentially Private Recurrent Language Models

1. (todo: Need to change sampling process) each user is selected independently with probability q, rather than always selecting a fixed number of users
2. enforce clipping of per-user updates so the total update has bounded L2 norm.
3. use different estimators for the average update (introduced next).
4. add Gaussian noise to the final average update.
"""

class DP_Clip(BaseDPFrame):
    """
    Differential Privacy mechanism with gradient clipping.

    Attributes:
        args: A namespace containing the configuration arguments for the mechanism.

    Methods:
        __init__(self, args): Initialize the DP_Clip mechanism.
        clip_local_update(self, local_grad, norm_type: float = 2.0): Clip local gradients.
        add_local_noise(self, local_grad: OrderedDict, extra_auxiliary_info: Any = None): Add local noise to gradients.
        add_global_noise(self, global_model: OrderedDict): Add global noise to the global model parameters.
        get_global_params(self): Get global parameters.
        compute_noise(self, size, qw): Compute noise.
        add_noise(self, w_global, qw): Add noise to global parameters.
    """

    def __init__(self, args):
        """
        Initialize the DP_Clip mechanism.

        Args:
            args: A namespace containing the configuration arguments for the mechanism.
        """
        super().__init__(args)
        self.clipping_norm = args.clipping_norm
        self.train_data_num_in_total = args.train_data_num_in_total
        self._scale = args.clipping_norm * args.noise_multiplier

    def clip_local_update(self, local_grad, norm_type: float = 2.0):
        """
        Clip local gradients.

        Args:
            local_grad (OrderedDict): Local gradients.
            norm_type (float): Type of norm to compute (default is 2.0).

        Returns:
            OrderedDict: Clipped local gradients.
        """
        total_norm = torch.norm(torch.stack(
            [torch.norm(local_grad[k], norm_type) for k in local_grad.keys()]), norm_type)
        clip_coef = self.clipping_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for k in local_grad.keys():
            local_grad[k].mul_(clip_coef_clamped)
        return local_grad

    def add_local_noise(self, local_grad: OrderedDict, extra_auxiliary_info: Any = None):
        """
        Add local noise to gradients.

        Args:
            local_grad (OrderedDict): Local gradients.
            extra_auxiliary_info (Any): Extra auxiliary information (not used).

        Returns:
            OrderedDict: Local gradients with added noise.
        """
        global_model_params = extra_auxiliary_info
        for k in global_model_params.keys():
            local_grad[k] = local_grad[k] - global_model_params[k]
        return self.clip_local_update(local_grad, self.clipping_norm)

    def add_global_noise(self, global_model: OrderedDict):
        """
        Add global noise to the global model parameters (not implemented).

        Args:
            global_model (OrderedDict): Global model parameters.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError(
            "add_global_noise method is not implemented.")

    def get_global_params(self):
        """
        Get global parameters (not implemented).

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError(
            "get_global_params method is not implemented.")

    def compute_noise(self, size, qw):
        """
        Compute noise for differential privacy.

        Args:
            size: Size of the noise.
            qw: Noise scaling factor.

        Returns:
            torch.Tensor: Noise tensor.
        """
        self._scale = self._scale / qw
        return torch.normal(mean=0, std=self._scale, size=size)

    def add_noise(self, w_global, qw):
        """
        Add noise to global parameters for differential privacy.

        Args:
            w_global (OrderedDict): Global model parameters.
            qw: Noise scaling factor.

        Returns:
            OrderedDict: Global model parameters with added noise.
        """
        new_params = OrderedDict()
        for k in w_global.keys():
            new_params[k] = self.compute_noise(
                w_global[k].shape, qw) + w_global[k]
        return new_params
