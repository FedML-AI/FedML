import argparse
from fedml.core.security.defense.cclip_defense import CClipDefense
from tests.security.defense.utils import create_fake_model_list, create_fake_global_w_local_w_MNIST


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file", "--cf", help="yaml configuration file", type=str, default="",
    )

    # default arguments
    parser.add_argument("--federated_optimizer", type=str, default="FedAvg")
    parser.add_argument("--tau", type=int, default=10)
    parser.add_argument("--bucket_size", type=int, default=3)
    args, unknown = parser.parse_known_args()
    return args


def test_before_aggregation():
    config = add_args()
    client_grad_list = create_fake_model_list(20)
    defense = CClipDefense(config)
    grad_list = defense.defend_before_aggregation(client_grad_list)
    print(f"grad_list = {grad_list}")


def test_after_aggregation():
    config = add_args()
    _, global_w = create_fake_global_w_local_w_MNIST()
    defense = CClipDefense(config)
    defense.initial_guess = global_w
    grad_list = defense.defend_after_aggregation(global_w)
    print(f"result = {grad_list}")


if __name__ == "__main__":
    test_before_aggregation()
    test_after_aggregation()
