import argparse

from ....security.defense.coordinate_wise_trimmed_mean_defense import (
    CoordinateWiseTrimmedMeanDefense,
)
from ...test.aggregation.aggregation_functions import AggregationFunction
from ....security.test.utils import create_fake_model_list


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    parser.add_argument("--beta", type=float, default=0.2)
    args, unknown = parser.parse_known_args()
    return args


def test_defense():
    client_grad_list = create_fake_model_list(20)
    config = add_args()
    defense = CoordinateWiseTrimmedMeanDefense(config)
    result = defense.run(
        client_grad_list, base_aggregation_func=AggregationFunction.FedAVG
    )
    print(f"result = {result}")


if __name__ == "__main__":
    test_defense()
