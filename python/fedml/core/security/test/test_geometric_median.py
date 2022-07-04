import torch

from fedml.core.security.defense.geometric_median import (
    GeometricMedian,
    _compute_middle_point,
    _compute_geometric_median,
)
from fedml.core.security.test.utils import create_fake_model_list


def _get_geometric_median_obj():
    return GeometricMedian(byzantine_client_num=2, client_num_per_round=20, batch_num=5)


def test_defense():
    gm = _get_geometric_median_obj()
    model_list = create_fake_model_list(gm.client_num_per_round)
    val = gm.defend(model_list)
    print(f"val={val}")


def test__compute_middle_point():
    alphas = [0.5, 0.5]
    batch_w = [
        torch.FloatTensor([[1, 0, 1], [2, 2, 2], [1, 1, 1]]),
        torch.FloatTensor([[1, 1, 1], [0, 0, 0], [1, 1, 1]]),
    ]
    print(f"middle_point = {_compute_middle_point(alphas, batch_w)}")


def test__compute_geometric_median():
    alphas = [1, 1, 1]
    batch_w = [
        torch.FloatTensor([[1, 0, 1], [2, 2, 2], [1, 1, 1]]),
        torch.FloatTensor([[1, 1, 1], [10, 10, 0], [1, 1, 1]]),
        torch.FloatTensor([[2, 2, 2], [2, 2, 2], [1, 2, 1]]),
    ]
    print(f"_compute_geometric_median = {_compute_geometric_median(alphas, batch_w)}")


if __name__ == "__main__":
    test_defense()
    # test__compute_middle_point()
    # test__compute_geometric_median()
