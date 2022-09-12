from typing import List, Tuple, Dict, Any
import torch
import numpy as np


class NbAFL_DP():
    def __init__(self, args):
        self.w_clip = args.w_clip  # c:clip
        self.total_round_num = args.total_round_num  # T
        self.nbafl_constant = args.nbafl_constant  # C:Constant
        self.num_train_data = args.num_train_data  # m

        self.client_num = args.client_num
        self.sample_client_num = args.sample_client_num
        self.nbafl_epsilon = args.nbafl_epsilon

        # set noise scale during upload
        self.nbafl_scale_u = args.w_clip * args.total_round_num * args.nbafl_constant \
                             / args.num_train_data / args.nbafl_epsilon

    def add_local_noise(self, raw_client_grad_list: List[Tuple[float, Dict]],
                        extra_auxiliary_info: Any = None):

        local_models = extra_auxiliary_info

        for parameters in local_models:
            for p in parameters:
                # Clip weight
                p.data = p.data / torch.max(torch.ones(size=p.shape),
                                            torch.abs(p.data) / self.w_clip)
                noise = torch.normal(mean=0, std=self.nbafl_scale_u, size=p.shape)
                p.data += noise

    def add_global_noise(
            self,
            global_model,
            extra_auxiliary_info: Any = None,
    ):
        # Clip weight
        for p in global_model.parameters():
            p.data = p.data / torch.max(torch.ones(size=p.shape),
                                        torch.abs(p.data) / self.w_clip)

        if len(self.sample_client_num) > 0:
            # Inject noise
            L = self.sample_client_num if self.sample_client_num > 0 else self.client_num
            if self.total_round_num > np.sqrt(self.client_num) * L:
                scale_d = 2 * self.w_clip * self.nbafl_constant * np.sqrt(
                    np.power(self.total_round_num, 2) -
                    np.power(L, 2) * self.client_num) / (
                                  min(self.sample_client_num.values()) * self.client_num *
                                  self.nbafl_epsilon)
                for p in global_model.parameters():
                    p.data += torch.normal(mean=0, std=scale_d, size=p.shape)
