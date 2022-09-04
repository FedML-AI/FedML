import argparse
from fedml.core.security.defense.norm_diff_clipping_defense import (
    NormDiffClippingDefense,
)
from fedml.ml.aggregator.agg_operator import FedMLAggOperator
from utils import create_fake_vectors, create_fake_model_list_MNIST


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
    parser.add_argument("--federated_optimizer", type=str, default="FedAvg")
    parser.add_argument("--norm_bound", type=float, default=5.0)
    args, unknown = parser.parse_known_args()
    return args


def test_norm_diff_clipping():
    print("--------test_norm_diff_clipping--------")
    config = add_args()
    defense = NormDiffClippingDefense(config)
    model_list = create_fake_model_list_MNIST(10)
    print(
        f"norm diff clipping result = {defense.run(model_list, base_aggregation_func=FedMLAggOperator.agg, extra_auxiliary_info=model_list[0][1])}"
    )


def test__get_clipped_norm_diff():
    print("--------test__get_clipped_norm_diff--------")
    config = add_args()
    defense = NormDiffClippingDefense(config)
    local_v, global_v = create_fake_vectors()
    print(f"local_v = {local_v} \nglobal_v = {global_v}")
    print(f"clipped weight diff = {defense._get_clipped_norm_diff(local_v, global_v)}")


if __name__ == "__main__":
    test_norm_diff_clipping()
    test__get_clipped_norm_diff()
