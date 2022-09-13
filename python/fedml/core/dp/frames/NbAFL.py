import torch
import numpy as np
from fedml.core.dp.frames.base_dp_solution import BaseDPFrame
from fedml.core.dp.mechanisms import Gaussian
from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism


class NbAFL_DP(BaseDPFrame):
    def __init__(self, args):
        super().__init__(args)
        self.set_ldp(
            DPMechanism(
                args.ldp_mechanism_type,
                args.ldp_epsilon,
                args.ldp_delta,
                args.ldp_sensitivity,
            )
        )

        self.w_clip = args.w_clip  # c:clip
        self.total_round_num = args.total_round_num  # T
        self.nbafl_constant = args.nbafl_constant  # C:Constant
        # self.num_train_data = args.num_train_data  # m
        self.sample_client_num = args.self.sample_client_num
        self.client_num = args.client_num
        # self.nbafl_epsilon = args.nbafl_epsilon

        # set noise scale during upload
        # self.nbafl_scale_u = args.w_clip * args.total_round_num * args.nbafl_constant \
        #                      / args.num_train_data / args.nbafl_epsilon

    def add_local_noise(self, local_grad: dict):
        for k in local_grad.keys():
            # Clip weight
            local_grad[k] = local_grad[k] / torch.max(torch.ones(size=local_grad[k].shape),
                                            torch.abs(local_grad[k]) / self.w_clip)
        return super().add_local_noise(local_grad=local_grad)

    def add_global_noise(self, global_model: dict):
        for k in global_model.keys():
            # Clip weight
            global_model[k] = global_model[k] / torch.max(torch.ones(size=global_model[k].shape),
                                            torch.abs(global_model[k]) / self.w_clip)
        # Inject noise
        if self.total_round_num > np.sqrt(self.client_num) * self.sample_client_num:
            scale_d = 2 * self.w_clip * self.nbafl_constant * np.sqrt(np.power(self.total_round_num, 2) -
                np.power(self.sample_client_num, 2) * self.client_num) / (self.sample_client_num * self.client_num * self.nbafl_epsilon)
            for k in global_model.keys():
                global_model[k] = Gaussian.compute_noise_using_sigma(scale_d, global_model[k].shape)
        return global_model



