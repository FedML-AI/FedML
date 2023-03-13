"""
ref: Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in neural information processing systems 32 (2019).
"""
import argparse
from collections import OrderedDict
from fedml.core.security.attack.dlg_attack import DLGAttack
from fedml.model.cv.resnet import resnet20, resnet56
from fedml.model.cv.resnet_all import resnet50


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file", "--cf", help="yaml configuration file", type=str, default="",
    )

    # default arguments
    parser.add_argument("--model", type=str, default="resnet56")
    parser.add_argument("--attack_round_num", type=int, default=200)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--training_type", type=str, default="simulation")
    parser.add_argument("--backend", type=str, default="sp")
    parser.add_argument("--device_type", type=str, default="cpu")
    parser.add_argument("--comm_round", type=int, default=1)
    parser.add_argument("--protected_layers", type=list, default=[54])

    args, unknown = parser.parse_known_args()
    return args


def test_attack_dlg():
    net = resnet56(10)
    args = add_args()
    attack = DLGAttack(args=args)
    attack.reconstruct_data_using_a_model(a_model=OrderedDict(net.state_dict()), extra_auxiliary_info=OrderedDict(net.state_dict()))


if __name__ == "__main__":
    test_attack_dlg()
