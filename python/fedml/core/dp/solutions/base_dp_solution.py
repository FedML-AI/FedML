from abc import ABC
from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism


class BaseDPSolution(ABC):
    def __init__(self, args=None):
        self.cdp = None
        self.ldp = None

    def set_cdp(self, dp_mechanism: DPMechanism):
        self.cdp = dp_mechanism

    def set_ldp(self, dp_mechanism: DPMechanism):
        self.ldp = dp_mechanism

    def is_local_dp(self):
        return self.ldp is not None

    def is_central_dp(self):
        return self.cdp is not None

    def before_add_local_noise(self, local_grad: dict):
        pass

    def add_local_noise(self, local_grad: dict):
        pass

    def after_add_local_noise(self, local_grad: dict):
        pass

    def before_add_global_noise(self, global_model: dict):
        pass

    def add_global_noise(self, global_model: dict):
        return self.cdp.add_noise(grad=global_model)

    def after_add_global_noise(self, global_model: dict):
        pass
