import argparse
from functools import reduce

from fedml.core.security.defense.residual_based_reweighting_defense import (
    ResidualBasedReweightingDefense,
)
from tests.security.defense.utils import create_fake_model_list

# def test_a():
#     model_list = create_fake_model_list(20)
#     w = model_list[0][1]
#     print(f"w = {w}")
#     for k in w.keys():
#         shape = w[k].shape
#         total_num = reduce(lambda x, y: x * y, shape)
#         print(f"total_num = {total_num}")





def add_args(mode = "gaussian"):
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    parser.add_argument("--mode", type=str, default=mode)
    args, unknown = parser.parse_known_args()
    return args


def test_defense_ResidualBasedReweighting_gaussian():
    model_list = create_fake_model_list(20)
    print(f"test krum")
    defend = ResidualBasedReweightingDefense(add_args("gaussian"))
    new_model_list = defend.defend_before_aggregation(model_list)
    print(f"model_list={new_model_list}")

def test_defense_ResidualBasedReweighting_median():
    model_list = create_fake_model_list(20)
    print(f"test krum")
    defend = ResidualBasedReweightingDefense(add_args("median"))
    new_model_list = defend.defend_before_aggregation(model_list)
    print(f"model_list={new_model_list}")

def test_defense_ResidualBasedReweighting_theilsen():
    model_list = create_fake_model_list(20)
    print(f"test krum")
    defend = ResidualBasedReweightingDefense(add_args("theilsen"))
    new_model_list = defend.defend_before_aggregation(model_list)
    print(f"model_list={new_model_list}")


if __name__ == "__main__":
    test_defense_ResidualBasedReweighting_gaussian()
    test_defense_ResidualBasedReweighting_median()
    test_defense_ResidualBasedReweighting_theilsen()
