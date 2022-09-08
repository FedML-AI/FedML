import argparse

import torch

from ..fed_privacy_mechanism import FedMLDifferentialPrivacy


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    # default arguments
    parser.add_argument("--enable_dp", type=bool, default=True)
    parser.add_argument("--mechanism_type", type=str, default="gaussian")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--dp_type", type=str, default="ldp")

    # epsilon, delta=0, sensitivity=1.0, mechanism_type="laplace"
    args, unknown = parser.parse_known_args()
    return args


def test_FedMLDifferentialPrivacy():
    print("----------- test_FedMLDifferentialPrivacy -----------")
    FedMLDifferentialPrivacy.get_instance().init(add_args())
    a_local_w = dict()
    a_local_w["linear.weight"] = torch.FloatTensor(
        [[0.1, 0.2, 0.2, 0.1], [0.15, 0.12, 0.02, 0.2], [0.3, 0.01, 0.21, 0.11]]
    )
    a_local_w["linear.bias"] = torch.FloatTensor([0.01, 0.19, 0.21])
    k = "linear.weight"
    print(f"grad = {a_local_w}")
    print(FedMLDifferentialPrivacy.get_instance().add_noise(a_local_w))


def test_add_noise_with_data_distribution():
    print("----------- test_add_noise_with_data_distribution -----------")
    FedMLDifferentialPrivacy.get_instance().init(add_args())
    a_local_w = dict()
    a_local_w["linear.weight"] = torch.FloatTensor(
        [[0.1, 0.2, 0.2, 0.1], [0.15, 0.12, 0.02, 0.2], [0.3, 0.01, 0.21, 0.11]]
    )
    a_local_w["linear.bias"] = torch.FloatTensor([0.01, 0.19, 0.21])
    k = "linear.weight"
    print(f"grad = {a_local_w}")
    print(
        FedMLDifferentialPrivacy.get_instance().add_noise_with_data_distribution(
            a_local_w
        )
    )


if __name__ == "__main__":
    test_FedMLDifferentialPrivacy()
    test_add_noise_with_data_distribution()
