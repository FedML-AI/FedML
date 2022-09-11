from abc import ABC
from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism


class BaseDPFrame(ABC):
    def __init__(self, args=None):
        self.cdp = None
        self.ldp = None

    def set_cdp(self, dp_mechanism: DPMechanism):
        self.cdp = dp_mechanism

    def set_ldp(self, dp_mechanism: DPMechanism):
        self.ldp = dp_mechanism

    def add_local_noise(self, local_grad: dict):
        return self.ldp.add_noise(grad=local_grad)

    def add_global_noise(self, global_model: dict):
        return self.cdp.add_noise(grad=global_model)