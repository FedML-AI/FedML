from abc import ABC
from collections import OrderedDict
import torch
from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism
from typing import List, Tuple


class BaseDPFrame(ABC):
    """
    Abstract base class for Differential Privacy mechanisms.

    Attributes:
        cdp: A DPMechanism instance for global differential privacy.
        ldp: A DPMechanism instance for local differential privacy.
        args: A namespace containing the configuration arguments for the mechanism.
        is_rdp_accountant_enabled: A boolean indicating whether RDP accountant is enabled.
        max_grad_norm: Maximum gradient norm for gradient clipping.

    Methods:
        __init__(self, args=None): Initialize the BaseDPFrame instance.
        set_cdp(self, dp_mechanism: DPMechanism): Set the global differential privacy mechanism.
        set_ldp(self, dp_mechanism: DPMechanism): Set the local differential privacy mechanism.
        add_local_noise(self, local_grad: OrderedDict): Add local noise to local gradients.
        add_global_noise(self, global_model: OrderedDict): Add global noise to global model parameters.
        set_params_for_dp(self, raw_client_model_or_grad_list): Set parameters for differential privacy mechanism.
        get_rdp_accountant_val(self): Get the differential privacy parameter for RDP accountant.
        global_clip(self, raw_client_model_or_grad_list): Apply gradient clipping to global gradients.
    """

    def __init__(self, args=None):
        """
        Initialize the BaseDPFrame instance.

        Args:
            args: A namespace containing the configuration arguments for the mechanism.
        """
        self.cdp = None
        self.ldp = None
        self.args = args
        self.is_rdp_accountant_enabled = False
        if hasattr(args, "max_grad_norm") and args.max_grad_norm is not None:
            self.max_grad_norm = args.max_grad_norm
        else:
            self.max_grad_norm = None

    def set_cdp(self, dp_mechanism: DPMechanism):
        """
        Set the global differential privacy mechanism.

        Args:
            dp_mechanism (DPMechanism): A DPMechanism instance for global differential privacy.
        """
        self.cdp = dp_mechanism

    def set_ldp(self, dp_mechanism: DPMechanism):
        """
        Set the local differential privacy mechanism.

        Args:
            dp_mechanism (DPMechanism): A DPMechanism instance for local differential privacy.
        """
        self.ldp = dp_mechanism

    @abstractmethod
    def add_local_noise(self, local_grad: OrderedDict):
        """
        Add local noise to local gradients.

        Args:
            local_grad (OrderedDict): Local gradients.

        Returns:
            OrderedDict: Local gradients with added noise.
        """
        pass

    @abstractmethod
    def add_global_noise(self, global_model: OrderedDict):
        """
        Add global noise to global model parameters.

        Args:
            global_model (OrderedDict): Global model parameters.

        Returns:
            OrderedDict: Global model parameters with added noise.
        """
        pass

    @abstractmethod
    def set_params_for_dp(self, raw_client_model_or_grad_list):
        """
        Set parameters for differential privacy mechanism.

        Args:
            raw_client_model_or_grad_list: List of raw client models or gradients.
        """
        pass

    def get_rdp_accountant_val(self):
        """
        Get the differential privacy parameter for RDP accountant.

        Returns:
            float: Differential privacy parameter.
        """
        if self.cdp is not None:
            dp_param = self.cdp.get_rdp_scale()
        elif self.ldp is not None:
            dp_param = self.ldp.get_rdp_scale()
        else:
            raise Exception("can not create rdp accountant")
        return dp_param

    def global_clip(self, raw_client_model_or_grad_list):
        """
        Apply gradient clipping to global gradients.

        Args:
            raw_client_model_or_grad_list: List of raw client models or gradients.

        Returns:
            List: List of clipped client models or gradients.
        """
        if self.max_grad_norm is None:
            return raw_client_model_or_grad_list
        new_grad_list = []
        for (num, local_grad) in raw_client_model_or_grad_list:
            for k in local_grad.keys():
                total_norm = torch.norm(torch.stack([torch.norm(local_grad[k], 2.0) for k in local_grad.keys()]),
                                        2.0)
                clip_coef = self.max_grad_norm / (total_norm + 1e-6)
                clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
                for k in local_grad.keys():
                    local_grad[k].mul_(clip_coef_clamped)
            new_grad_list.append((num, local_grad))
        return new_grad_list
