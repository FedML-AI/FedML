import argparse
from fedml.core.security.defense.coordinate_wise_trimmed_mean_defense import (
    CoordinateWiseTrimmedMeanDefense,
)
from utils import create_fake_model_list


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    parser.add_argument("--federated_optimizer", type=str, default="FedAvg")
    parser.add_argument("--beta", type=float, default=0.2)
    args, unknown = parser.parse_known_args()
    return args


def test_defense():
    print("-------------CoordinateWiseTrimmedMeanDefense-------------")
    client_grad_list = create_fake_model_list(5)
    print(f"client_grad_list={client_grad_list}")
    defense = CoordinateWiseTrimmedMeanDefense(add_args())
    grads = defense.defend_before_aggregation(client_grad_list)
    print(f"grads = {grads}")


if __name__ == "__main__":
    test_defense()
