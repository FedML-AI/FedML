import torch
import numpy as np

from .defense_base import BaseDefenseMethod

"""
defense @ client, added by Kai, 07/10/2022
ref: Sun, Jingwei, et al. "Provable defense against privacy leakage in federated learning from representation perspective." 
arXiv preprint arXiv:2012.06043 (2020).
added by Kai, 07/10/2022

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
        criterion,
        defense_data,
        attack_method,
        defense_method,
        defense_label=84,
    ):
        self.num_class = num_class  # number of classess of the dataset
        self.model = model
        self.criterion = criterion
        self.defense_data = defense_data
        self.defense_label = defense_label
        self.attack_method = attack_method
        self.defense_method = defense_method

    # FIX: incorrect output with soteria and model compression
    def defend(self, local_w, global_w, refs=None):
        # load local model
        self.model.load_state_dict(local_w, strict=True)
        original_dy_dx = refs
        self.defense_data.requires_grad = True

        if self.attack_method == "dlg":
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

            y = self.criterion(out, gt_onehot_label)
            dy_dx = torch.autograd.grad(y, self.model.parameters())

            # share the gradients with other clients
            defensed_original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            if self.defense_method == "soteria":
                defensed_original_dy_dx[8] = defensed_original_dy_dx[8] * torch.Tensor(
                    mask
                )

            elif self.defense_method == "model compression":
                for i in range(len(defensed_original_dy_dx)):
                    grad_tensor = defensed_original_dy_dx[i].cpu().numpy()
                    flattened_weights = np.abs(grad_tensor.flatten())
                    # Generate the pruning threshold according to 'prune by percentage'. (Your code: 1 Line)
                    thresh = np.percentile(flattened_weights, 10)
                    grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
                    defensed_original_dy_dx[i] = torch.Tensor(grad_tensor)

            elif self.defense_method == "differential privacy":
                for i in range(len(defensed_original_dy_dx)):
                    grad_tensor = defensed_original_dy_dx[i].cpu().numpy()
                    noise = np.random.laplace(0, 1e-1, size=grad_tensor.shape)
                    grad_tensor = grad_tensor + noise
                    defensed_original_dy_dx[i] = torch.Tensor(grad_tensor)

        return original_dy_dx[0][0][0][0], defensed_original_dy_dx[0][0][0][0]

    def label_to_onehot(self, target, num_classes=100):
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target
