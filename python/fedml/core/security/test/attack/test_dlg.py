"""
ref: Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in neural information processing systems 32 (2019).
added by Kai, 07/06/2022
"""

# test using local directory
# import os, sys
# __file__ = '/Users/kai/Documents/FedML/python/fedml/core/'
# sys.path.append(__file__)
# from security.attack.dlg_attack import DLGAttack
# from security.test.utils import create_fake_gradient_Cifar100, create_fake_model_Cifar100

from fedml.core.security.attack.dlg_attack import DLGAttack
from fedml.core.security.test.utils import (
    create_fake_gradient_Cifar100,
    create_fake_model_Cifar100,
)

import torch
import torch.nn.functional as F
import torch.nn as nn

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
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def test__attack_dlg():
    # local_gradient -> which could be inferred via w = w - eta * g
    local_gradient = create_fake_gradient_Cifar100()
    local_w = create_fake_model_Cifar100()
    attack = DLGAttack(
        data_size=[3, 32, 32],
        num_class=100,
        model=LeNet(),
        criterion=cross_entropy_for_onehot,
        attack_epoch=100,
    )
    attack.attack(local_w, global_w=None, refs=local_gradient)


if __name__ == "__main__":
    test__attack_dlg()
