import torch
from fedml.core.security.common.utils import compute_middle_point
from fedml.core.security.defense.geometric_median_defense import GeometricMedianDefense
from fedml.core.security.test.utils import create_fake_model_list


def _get_geometric_median_obj():
    return GeometricMedianDefense(
        byzantine_client_num=2, client_num_per_round=20, batch_num=5
    )


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


def test__compute_geometric_median():
    alphas = [0.3, 0.3, 0.4]
    batch_w = [
        torch.FloatTensor([[1, 0, 1], [2, 2, 2], [1, 1, 1]]),
        torch.FloatTensor([[1, 1, 1], [10, 10, 0], [1, 1, 1]]),
        torch.FloatTensor([[2, 2, 2], [2, 2, 2], [1, 2, 1]]),
    ]
    print(
        f"_compute_geometric_median = {GeometricMedianDefense._compute_geometric_median(alphas, batch_w)}"
    )


if __name__ == "__main__":
    test_defense()
    test__compute_middle_point()
    test__compute_geometric_median()
