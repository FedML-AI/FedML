import argparse

from ....security.defense.dp import DifferentialPrivacy
from ....security.test.aggregation.aggregation_functions import AggregationFunction
from ....security.test.utils import create_fake_model_list


def add_args(dp_type):
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    parser.add_argument("--dp_type", type=str, default=dp_type)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=1)
    parser.add_argument("--sensitivity", type=int, default=1)
    parser.add_argument("--mechanism_type", type=str, default="gaussian")
    args, unknown = parser.parse_known_args()
    return args


def test_cdp():
    print("-----test defense-----")
    config = add_args("cdp")
    model_list = create_fake_model_list(10)
    print(f"model_list = {model_list}")
    dp = DifferentialPrivacy(config)
    res = dp.run(model_list, base_aggregation_func=AggregationFunction.FedAVG)
    print(f"cdp result = {res}")

def test_ldp():
    print("-----test defense-----")
    config = add_args("ldp")
    model_list = create_fake_model_list(10)
    print(f"model_list = {model_list}")
    dp = DifferentialPrivacy(config)
    res = dp.run(model_list, base_aggregation_func=AggregationFunction.FedAVG)
    print(f"ldp result = {res}")


if __name__ == '__main__':
    test_cdp()
    test_ldp()