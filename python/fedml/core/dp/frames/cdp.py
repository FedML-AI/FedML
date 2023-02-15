import logging
from collections import OrderedDict
from fedml.core.dp.budget_accountant.rdp_accountant import RDP_Accountant
from fedml.core.dp.frames.base_dp_solution import BaseDPFrame
from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism


class GlobalDP(BaseDPFrame):
    def __init__(self, args):
        super().__init__(args)
        self.set_cdp(DPMechanism(args.mechanism_type, args.epsilon, args.delta, args.sensitivity))
        self.enable_rdp_accountant = False
        if hasattr(args, "enable_rdp_accountant") and args.enable_rdp_accountant:
            self.is_rdp_accountant_enabled = True
            self.sample_rate = args.client_num_per_round / args.client_num_in_total
            self.accountant = RDP_Accountant(alpha=args.rdp_alpha, dp_mechanism=args.mechanism_type, args=args)

    def add_global_noise(self, global_model: OrderedDict):
        if self.is_rdp_accountant_enabled:
            self.accountant.step(noise_multiplier=self.cdp.get_rdp_scale(), sample_rate=self.sample_rate) # todo: ask???
        return super().add_global_noise(global_model=global_model)