import argparse

from fedml.core.security.defense.cclip_defense import CClipDefense
from fedml.core.security.test.aggregation.aggregation_functions import AggregationFunction
from fedml.core.security.test.utils import create_fake_model_list


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
    parser.add_argument("--tau", type=int, default=10)

    parser.add_argument("--bucket_size", type=int, default=3)

    args, unknown = parser.parse_known_args()
    return args


def test_defense(config):
    client_grad_list = create_fake_model_list(20)
    cclip = CClipDefense(config)
    result = cclip.run(client_grad_list, global_model=None, base_aggregation_func=AggregationFunction.FedAVG)
    print(f"result = {result}")


if __name__ == "__main__":
    args = add_args()
    test_defense(args)
