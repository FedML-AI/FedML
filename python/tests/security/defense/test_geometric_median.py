import argparse
import torch
from fedml.core.security.common.utils import compute_middle_point, compute_geometric_median
from fedml.core.security.defense.geometric_median_defense import GeometricMedianDefense
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
    parser.add_argument("--byzantine_client_num", type=int, default=2)
    parser.add_argument("--client_num_per_round", type=int, default=20)
    parser.add_argument("--batch_num", type=int, default=5)
    args, unknown = parser.parse_known_args()
    return args


def _get_geometric_median_obj():
    config = add_args()
    return GeometricMedianDefense(config)


def test_defense():
    print("-----test defense-----")
    gm = _get_geometric_median_obj()
    model_list = create_fake_model_list(gm.client_num_per_round)
    res = gm.run(model_list)
    print(f"aggregation result = {res}")


def test__compute_middle_point():
    alphas = [0.5, 0.5]
    batch_w = [
        torch.FloatTensor([[1, 0, 1], [2, 2, 2], [1, 1, 1]]),
        torch.FloatTensor([[1, 1, 1], [0, 0, 0], [1, 1, 1]]),
    ]
    print(f"middle_point = {compute_middle_point(alphas, batch_w)}")


def test_compute_geometric_median():
    alphas = [0.3, 0.3, 0.4]
    batch_w = [
        torch.FloatTensor([[1, 0, 1], [2, 2, 2], [1, 1, 1]]),
        torch.FloatTensor([[1, 1, 1], [10, 10, 0], [1, 1, 1]]),
        torch.FloatTensor([[2, 2, 2], [2, 2, 2], [1, 2, 1]]),
    ]
    print(
        f"_compute_geometric_median = {compute_geometric_median(alphas, batch_w)}"
    )


if __name__ == "__main__":
    test_defense()
    test__compute_middle_point()
    test_compute_geometric_median()
