from abc import ABC
from collections import OrderedDict

from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism
import torch


class BaseDPFrame(ABC):
    def __init__(self, args=None):
        self.cdp = None
        self.ldp = None
        self.args = args

    def set_cdp(self, dp_mechanism: DPMechanism):
        self.cdp = dp_mechanism

    def set_ldp(self, dp_mechanism: DPMechanism):
        self.ldp = dp_mechanism

    def add_local_noise(self, local_grad: OrderedDict):
        return self.ldp.add_noise(grad=local_grad)

    def add_global_noise(self, global_model: OrderedDict):
        return self.cdp.add_noise(grad=global_model)


