"""
ref: Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in neural information processing systems 32 (2019).
"""

# test using local directory
import os, sys

__file__ = "/Users/kai/Documents/FedML/python/"
sys.path.append(__file__)

from fedml.core.security.attack.dlg_attack import DLGAttack
from fedml.core.security.common.utils import cross_entropy_for_onehot, label_to_onehot
from fedml.core.security.common.attack_defense_data_loader import (
    AttackDefenseDataLoader,
)

import torch
import torch.nn.functional as F
import torch.nn as nn


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
        self.fc = nn.Sequential(nn.Linear(768, 10))

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def test__attack_dlg(attack_client_idx, net, attack_epoch=5):
    attack = DLGAttack(attack_client_idx, net, attack_epoch,)
    # train_data_local_dict -> 5, attack_client_idx -> 0
    data = dataset[5][attack_client_idx].dataset
    img, label = data[0]  # the first image
    data_size = [1] + list(img.size())
    num_class = dataset[7]
    refs = (data_size, label, dataset[7])

    # compute original gradient (simulation, thie operation should be implemeted locally in the real application)
    gt_label = torch.Tensor([label]).long()
    gt_label = gt_label.view(1,)
    gt_onehot_label = label_to_onehot(gt_label, num_class)
    img = img.view(1, *img.size()).requires_grad_()
    out = net(img)
    y = cross_entropy_for_onehot(out, gt_onehot_label)
    local_w = torch.autograd.grad(y, net.parameters())

    # this is a reconstruction attack, I think there should be a new abstract method (e.g. recon_data)rather than attack_model or poison_data
    attack.attack_model(local_w=local_w, global_w=None, refs=refs)


if __name__ == "__main__":
    client_num = 3
    batch_size = 32
    dataset = AttackDefenseDataLoader.load_cifar10_data(
        client_num=client_num, batch_size=batch_size
    )
    test__attack_dlg(0, LeNet())
