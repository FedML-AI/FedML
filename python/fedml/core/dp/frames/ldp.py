from collections import OrderedDict

from fedml.core.dp.frames.base_dp_solution import BaseDPFrame
from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism


class LocalDP(BaseDPFrame):
    """
    Local Differential Privacy mechanism.

    Attributes:
        args: A namespace containing the configuration arguments for the mechanism.

    Methods:
        __init__(self, args): Initialize the LocalDP mechanism.
        add_local_noise(self, local_grad: OrderedDict): Add local noise to the gradients.
    """

    def __init__(self, args):
        """
        Initialize the LocalDP mechanism.

        Args:
            args: A namespace containing the configuration arguments for the mechanism.
        """
        super().__init__(args)
        self.set_ldp(DPMechanism(args.mechanism_type,
                     args.epsilon, args.delta, args.sensitivity))

    def add_local_noise(self, local_grad: OrderedDict):
        """
        Add local noise to the gradients.

        Args:
            local_grad (OrderedDict): Local gradients.

        Returns:
            OrderedDict: Local gradients with added noise.
        """
        return super().add_local_noise(local_grad=local_grad)
