"""
ref: Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in neural information processing systems 32 (2019).
"""
import argparse

import torch

from ...attack.dlg_attack import DLGAttack
from ...common.attack_defense_data_loader import AttackDefenseDataLoader
from ...common.net import LeNet
from ...common.utils import cross_entropy_for_onehot, label_to_onehot


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file", "--cf", help="yaml configuration file", type=str, default="",
    )

    # default arguments
    parser.add_argument("--model", type=str, default="LeNet")
    parser.add_argument("--attack_epoch", type=int, default=5)

    args, unknown = parser.parse_known_args()
    return args


def test__attack_dlg():
    dataset = AttackDefenseDataLoader.load_cifar10_data(client_num=3, batch_size=32)
    net = LeNet()
    attack = DLGAttack(net, attack_epoch=5)
    # train_data_local_dict -> 5, attack_client_idx -> 0
    attack_client_idx = 0

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
    attack.reconstruct(local_w=local_w, global_w=None, refs=refs)


if __name__ == "__main__":
    test__attack_dlg()
