from fedml.core.security.defense.coordinate_wise_median_defense import CoordinateWiseMedianDefense
from utils import create_fake_model_list


def test_defense():
    client_grad_list = create_fake_model_list(20)
    defense = CoordinateWiseMedianDefense(config=None)
    result = defense.run(client_grad_list)
    print(f"result = {result}")


if __name__ == "__main__":
    test_defense()
