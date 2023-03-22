from fedml.core.security.defense.RFA_defense import RFADefense
from tests.security.defense.utils import create_fake_model_list


def test_defense():
    client_grad_list = create_fake_model_list(5)
    defense = RFADefense(config=None)
    result = defense.defend_on_aggregation(client_grad_list)
    print(f"result = {result}")


if __name__ == "__main__":
    test_defense()