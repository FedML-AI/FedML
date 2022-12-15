import argparse
from collections import OrderedDict

import torch

from ..fedml_differential_privacy import FedMLDifferentialPrivacy

a_local_w = OrderedDict()
a_local_w["linear.weight"] = torch.FloatTensor(
    [[0.1, 0.2, 0.2, 0.1], [0.15, 0.12, 0.02, 0.2], [0.3, 0.01, 0.21, 0.11]]
)
a_local_w["linear.bias"] = torch.FloatTensor([0.01, 0.19, 0.21])


def add_gaussian_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    parser.add_argument("--enable_dp", type=bool, default=True)
    parser.add_argument("--mechanism_type", type=str, default="gaussian")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--dp_type", type=str, default="ldp")

    # epsilon, delta=0, sensitivity=1.0, mechanism_type="laplace"
    args, unknown = parser.parse_known_args()
    return args


def add_laplace_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    parser.add_argument("--enable_dp", type=bool, default=True)
    parser.add_argument("--mechanism_type", type=str, default="laplace")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--dp_type", type=str, default="ldp")
    args, unknown = parser.parse_known_args()
    return args


def test_FedMLDifferentialPrivacy_gaussian():
    print("----------- test_FedMLDifferentialPrivacy - gaussian mechanism -----------")
    FedMLDifferentialPrivacy.get_instance().init(add_gaussian_args())
    print(f"grad = {a_local_w}")
    print(FedMLDifferentialPrivacy.get_instance().add_noise(a_local_w))


def test_FedMLDifferentialPrivacy_laplace():
    print("----------- test_FedMLDifferentialPrivacy - laplace mechanism -----------")
    FedMLDifferentialPrivacy.get_instance().init(add_laplace_args())
    print(f"grad = {a_local_w}")
    print(FedMLDifferentialPrivacy.get_instance().add_noise(a_local_w))


if __name__ == "__main__":
    test_FedMLDifferentialPrivacy_gaussian()
    test_FedMLDifferentialPrivacy_laplace()
