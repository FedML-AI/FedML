import numpy as np 
from fedml.core.security.test.utils import create_fake_model_list
from fedml.core.security.defense.bulyan_defense import BulyanDefense


def _get_bulyan_obj():
    return BulyanDefense(byzantine_client_num=1, client_num_per_round=8)


def test_defense():
    mk = _get_bulyan_obj()
    model_list = create_fake_model_list(mk.client_num_per_round)
    val = mk.defend(model_list)
    print(f"val={val}")

def test__compute_middle_point():
    by = _get_bulyan_obj()
    select_indexs, selected_set ,agg_grads= by._bulyan(np.array([[-10, -20, -30], [5, 8, 11], [3, 8, 9], [5, 7, 9],
                                                   [5, 8, 9], [5, 8, 11], [3, 8, 9], [5, 7, 9]]), 8,1)
    print(f"{select_indexs, selected_set ,agg_grads}")


if __name__ == "__main__":
    test_defense()
    test__compute_middle_point()