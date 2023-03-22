import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from fedml.core.security.defense.soteria_defense import SoteriaDefense
from utils import create_fake_gradient_Cifar100, create_fake_model_Cifar100, create_fake_data_Cifar100

"""
TODO FIX: load model and the corresponding parameters from FedML system
"""

"""
ref: Sun, Jingwei, et al. "Provable defense against privacy leakage in federated learning from representation perspective." 
arXiv preprint arXiv:2012.06043 (2020).
added by Kai, 07/10/2022
"""


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(nn.Linear(768, 100))

    def forward(self, x):
        out = self.body(x)
        feature = out.view(out.size(0), -1)
        out = self.fc(feature)
        return out, feature


def test_defense_soteria_dlg():
    # local_gradient -> which could be inferred via w = w - eta * g
    local_gradient = create_fake_gradient_Cifar100()
    print(f"local = {local_gradient}")
    local_w = create_fake_model_Cifar100()
    local_data = create_fake_data_Cifar100()
    defense_methods = ["soteria"]
    # defense_methods = ["soteria", "model compression", "differential privacy"]

    # todo: fix format of local_w
    for med in defense_methods:
        defense = SoteriaDefense(num_class=100, model=LeNet(), defense_data=local_data,)
        def_grad = defense.run(
            raw_client_grad_list=local_w, base_aggregation_func=None, extra_auxiliary_info=local_gradient
        )
        logging.info(f"method = {med}, grad = {def_grad}")


if __name__ == "__main__":
    test_defense_soteria_dlg()
