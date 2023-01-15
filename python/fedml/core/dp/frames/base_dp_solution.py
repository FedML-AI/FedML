from abc import ABC
from collections import OrderedDict
from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism
from typing import List, Tuple


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

    def set_params_for_dp(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        pass

    def get_rdp_accountant_val(self):
        if self.cdp is not None:
            dp_param = self.cdp.get_rdp_scale()
        elif self.ldp is not None:
            dp_param = self.ldp.get_rdp_scale()
        else:
            raise Exception("can not create rdp accountant")
        return dp_param

