"""
ref: Sun, Jingwei, et al. "Provable defense against privacy leakage in federated learning from representation perspective." 
arXiv preprint arXiv:2012.06043 (2020).
added by Kai, 07/10/2022
"""

# test using local directory
# import os, sys
# __file__ = '/Users/kai/Documents/FedML/python/fedml/core/'
# sys.path.append(__file__)
# from security.defense.soteria_defense import SoteriaDefense
# from security.test.utils import create_fake_gradient_Cifar100, create_fake_model_Cifar100, create_fake_data_Cifar100

from fedml.core.security.defense.soteria_defense import SoteriaDefense
from fedml.core.security.test.utils import (
    create_fake_gradient_Cifar100,
    create_fake_model_Cifar100,
    create_fake_data_Cifar100,
)

import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

"""
TODO FIX: load model and the corresponding parameters from FedML system
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


def test__defense_soteria_dlg():
    # local_gradient -> which could be inferred via w = w - eta * g
    local_gradient = create_fake_gradient_Cifar100()
    local_w = create_fake_model_Cifar100()
    local_data = create_fake_data_Cifar100()
    defense_methods = ["soteria"]
    # defense_methods = ["soteria", "model compression", "differential privacy"]
    for med in defense_methods:
        defense = SoteriaDefense(
            num_class=100,
            model=LeNet(),
            defense_data=local_data,
            attack_method="dlg",
            defense_method=med,
        )
        ori_grad, def_grad = defense.defend(local_w, global_w=None, refs=local_gradient)
        logging.info(
            f"method = {med}, _original_gradient = {ori_grad}, _after_defend_gradient = {def_grad}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test__defense_soteria_dlg()
