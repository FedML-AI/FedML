"""
ref: Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in neural information processing systems 32 (2019).
"""
import argparse


from fedml.core.security.attack.dlg_attack import DLGAttack
from fedml.core.security.common.net import LeNet
from fedml.core.security.common.utils import cross_entropy_for_onehot, label_to_onehot
from fedml.core.security.common.attack_defense_data_loader import (
    AttackDefenseDataLoader,
)
import torch


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

    # todo: @ Kai, the format of local_w should be like:
    # {'linear.weight': tensor([[-0.0003, 0.0192, -0.0294, 0.0219, 0.0037, 0.0021],
    #                           [-0.0198, -0.0150, -0.0104, -0.0203, -0.0060, -0.0299],
    #                           [-0.0201, 0.0149, -0.0333, -0.0203, 0.0012, 0.0080],
    #                           [0.0237, 0.0103, -0.0219, 0.0088, -0.0009, 0.0009],
    #                           [0.0144, -0.0336, -0.0346, -0.0222, -0.0025, -0.0138],
    #                           [-0.0196, -0.0118, 0.0230, -0.0202, 0.0172, 0.0355]]),
    #  'linear.bias': tensor([-0.0745, -0.0578, -0.0899, -0.0662, 0.1122, 0.0295])}
    attack.reconstruct_data(a_gradient=local_w, extra_auxiliary_info=refs)


if __name__ == "__main__":
    test__attack_dlg()
