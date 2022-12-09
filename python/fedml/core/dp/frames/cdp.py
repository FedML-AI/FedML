from collections import OrderedDict

from fedml.core.dp.frames.base_dp_solution import BaseDPFrame
from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism


class GlobalDP(BaseDPFrame):
    def __init__(self, args):
        super().__init__(args)

        self.set_cdp(
            DPMechanism(args.mechanism_type, args.epsilon, args.delta, args.sensitivity, args)
        )

    def add_global_noise(self, w_locals, qw):
        return super().add_global_noise(w_locals, qw)

    def clip_local_update(self, update, clipping_norm):
        return super().clip_local_update(update, clipping_norm)

