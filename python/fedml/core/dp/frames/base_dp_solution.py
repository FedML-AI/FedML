from abc import ABC
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

    def add_local_noise(self, local_grad: dict):
        return self.ldp.add_noise(grad=local_grad)

    def add_global_noise(self, w_locals, qw):
        return self.cdp.add_noise(w_locals, qw)

    def clip_local_update(self, update, clipping_norm):
        return self.cdp.clip_local_update(update, clipping_norm)
