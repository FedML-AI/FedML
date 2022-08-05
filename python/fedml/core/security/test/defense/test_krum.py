import argparse

from fedml.core.security.defense.krum_defense import KrumDefense
from fedml.core.security.test.aggregation.aggregation_functions import (
    AggregationFunction,
)
from fedml.core.security.test.utils import create_fake_model_list


def add_args_krum():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    # default arguments
    parser.add_argument("--byzantine_client_num", type=int, default=2)
    parser.add_argument("--multi", type=bool, default=False)
    args, unknown = parser.parse_known_args()
    return args


def add_args_multi_krum():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    # default arguments
    parser.add_argument("--byzantine_client_num", type=int, default=2)
    parser.add_argument("--multi", type=bool, default=True)
    args, unknown = parser.parse_known_args()
    return args


def test_defense_Krum():
    model_list = create_fake_model_list(20)
    print(f"test krum")
    krum = KrumDefense(add_args_krum())
    filtered_model_list = krum.run(
        model_list, base_aggregation_func=AggregationFunction.FedAVG
    )
    print(f"filtered_model_list={filtered_model_list}")


def test_defense_multi_Krum():
    model_list = create_fake_model_list(20)
    print(f"test multi-krum")
    krum = KrumDefense(add_args_multi_krum())
    filtered_model_list = krum.run(
        model_list, base_aggregation_func=AggregationFunction.FedAVG
    )
    print(f"filtered_model_list={filtered_model_list}")


if __name__ == "__main__":
    test_defense_Krum()
    test_defense_multi_Krum()
