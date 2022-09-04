import argparse
import numpy as np
from fedml.core.security.defense.bulyan_defense import BulyanDefense
from fedml.ml.aggregator.agg_operator import FedMLAggOperator
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

    # default arguments
    parser.add_argument("--federated_optimizer", type=str, default="FedAvg")
    parser.add_argument("--byzantine_client_num", type=int, default=1)
    parser.add_argument("--client_num_per_round", type=int, default=8)

    args, unknown = parser.parse_known_args()
    return args


def test_defense():
    config = add_args()
    mk = BulyanDefense(config)
    model_list = create_fake_model_list(mk.client_num_per_round)
    val = mk.run(model_list, FedMLAggOperator.agg)
    print(f"val={val}")


def test__compute_middle_point():
    config = add_args()
    by = BulyanDefense(config)
    select_indexs, selected_set, agg_grads = by._bulyan(
        np.array(
            [
                [-10, -20, -30],
                [5, 8, 11],
                [3, 8, 9],
                [5, 7, 9],
                [5, 8, 9],
                [5, 8, 11],
                [3, 8, 9],
                [5, 7, 9],
            ]
        ),
        8,
        1,
    )
    print(f"{select_indexs, selected_set ,agg_grads}")


if __name__ == "__main__":
    test_defense()
    test__compute_middle_point()
