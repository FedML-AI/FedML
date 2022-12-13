import logging
from collections import OrderedDict
from fedml.core.dp.frames.base_dp_solution import BaseDPFrame
from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism


class GlobalDP(BaseDPFrame):
    def __init__(self, args):
        super().__init__(args)

        self.set_cdp(
            DPMechanism(args.mechanism_type, args.epsilon, args.delta, args.sensitivity)
        )

    def add_global_noise(self, global_model: OrderedDict):
        return super().add_global_noise(global_model=global_model)