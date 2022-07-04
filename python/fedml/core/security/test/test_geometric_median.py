from fedml.core.security.defense.geometric_median import GeometricMedian
from fedml.core.security.test.utils import create_fake_model_list


def _get_geometric_median_obj():
    return GeometricMedian(byzantine_client_num=2, client_num_per_round=20, batch_num=5)


def test_defense():
    gm = _get_geometric_median_obj()
    model_list = create_fake_model_list(gm.client_num_per_round)
    val = gm.defend(model_list)
    print(val)



if __name__ == '__main__':
    test_defense()
