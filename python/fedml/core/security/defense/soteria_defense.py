from collections import OrderedDict
import torch
import numpy as np
from .defense_base import BaseDefenseMethod
from ..common.utils import cross_entropy_for_onehot
from typing import Callable, List, Tuple, Dict, Any

"""
defense @ client, added by Kai, 07/10/2022
ref: Sun, Jingwei, et al. "Provable defense against privacy leakage in federated learning from representation perspective." 
https://arxiv.org/pdf/2012.06043
added by Kai, 07/10/2022

Consider a malicious server; add a perturbation to data representation; 
the reconstructed data using the perturbed data representations are as dissimilar as possible to the raw data. 

Experiment setting in paper: FedAVG; defend against DLG attack GS attack; MNIST, CIFAR10


TODO
Steps:
(1)
(2)
(3)
"""


class SoteriaDefense(BaseDefenseMethod):
    def __init__(
        self,
        num_class,
        model,
        defense_data,
        defense_label=84,
    ):
        self.num_class = num_class  # number of classess of the dataset
        self.model = model
        self.defense_data = defense_data
        self.defense_label = defense_label

    # FIX: incorrect output with soteria and model compression
    # def defend(self, local_w, global_w, refs=None):
    def run(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            base_aggregation_func: Callable = None,
            extra_auxiliary_info: Any = None,
    ) -> Dict:
        # load local model
        self.model.load_state_dict(raw_client_grad_list, strict=True)
        original_dy_dx = extra_auxiliary_info # refs for local gradient
        self.defense_data.requires_grad = True

        gt_onehot_label = self.label_to_onehot(
            torch.Tensor([self.defense_label]).long(), num_classes=self.num_class
        )

        # compute ||dr/dX||/||r||
        out, feature_fc1_graph = self.model(self.defense_data)
        deviation_f1_target = torch.zeros_like(feature_fc1_graph)
        deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
        for f in range(deviation_f1_x_norm.size(1)):
            deviation_f1_target[:, f] = 1
            feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
            deviation_f1_x = self.defense_data.grad.data
            deviation_f1_x_norm[:, f] = torch.norm(
                deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1
            ) / (feature_fc1_graph.data[:, f])
            self.model.zero_grad()
            self.defense_data.grad.data.zero_()
            deviation_f1_target[:, f] = 0

        # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
        deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
        thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), 1)
        mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(
            np.float32
        )

        y = cross_entropy_for_onehot(out, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, self.model.parameters())

        # share the gradients with other clients
        defensed_original_dy_dx = list((_.detach().clone() for _ in dy_dx))

        defensed_original_dy_dx[8] = defensed_original_dy_dx[8] * torch.Tensor(mask)
        aggregation_result = defensed_original_dy_dx[0][0][0][0]
        print(
            f"_original_gradient = {original_dy_dx[0][0][0][0]}, _after_defend_gradient = {aggregation_result}"
        )

        return aggregation_result

    def label_to_onehot(self, target, num_classes=100):
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target
