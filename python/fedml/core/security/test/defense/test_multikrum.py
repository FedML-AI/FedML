import numpy as np 
import torch
from fedml.core.security.test.utils import create_fake_model_list
from fedml.core.security.defense.multikrum_defense import MultiKrumDefense


def _get_multikrum_obj():
    return MultiKrumDefense(byzantine_client_num=1, client_num_per_round=5)


def test_defense():
    mk = _get_multikrum_obj()
    model_list = create_fake_model_list(mk.client_num_per_round)
    val = mk.defend(model_list)
    print(f"val={val}")

def test__compute_middle_point():
    mk = _get_multikrum_obj()
    alpha, multikrum_avg, scores= mk._multi_krum(np.array([[1, 2, 3], [5, 8, 11], [3, 8, 9], [5, 7, 9], [5, 8, 9]]), 5,1)
    print(f"results = {alpha, multikrum_avg, scores}")


if __name__ == "__main__":
    test_defense()
    test__compute_middle_point()
