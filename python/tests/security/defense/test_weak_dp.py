import argparse

from fedml.core.security.defense.weak_dp_defense import WeakDPDefense
from fedml.ml.aggregator.agg_operator import FedMLAggOperator
from utils import create_fake_model_list_MNIST


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file", "--cf", help="yaml configuration file", type=str, default="",
    )
    # default arguments
    parser.add_argument("--federated_optimizer", type=str, default="FedAvg")
    parser.add_argument("--stddev", type=float, default=0.1)
    args, unknown = parser.parse_known_args()
    return args


def test_weak_dp():
    print("--------test_weak dp--------")
    config = add_args()
    defense = WeakDPDefense(config)
    model_list = create_fake_model_list_MNIST(10)
    print(f"original aggregation result = {FedMLAggOperator.agg}")
    print(f"weak dp result = {defense.run(model_list, base_aggregation_func=FedMLAggOperator.agg)}")


def test_add_noise():
    print("--------test_weak dp--------")
    config = add_args()
    defense = WeakDPDefense(config)
    model_list = create_fake_model_list_MNIST(10)
    print(f"original weight = {model_list[0][1]}")
    print(f"weak dp result = {defense._add_noise(model_list[0][1])}")


if __name__ == "__main__":
    test_weak_dp()
    test_add_noise()
