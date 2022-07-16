from fedml.core.security.defense.cclip_defense import CClipDefense
from fedml.core.security.test.utils import create_fake_model_list, normal_aggregation


def test_defense():
    client_grad_list = create_fake_model_list(20)
    cclip = CClipDefense(tau=10)
    new_grad_list = cclip.defend(client_grad_list, global_w=None)
    print(f"new_grad_list={new_grad_list}")


def test_robustify_global_model():
    client_grad_list = create_fake_model_list(20)
    avg_params = normal_aggregation(client_grad_list)
    cclip = CClipDefense(tau=10)
    cclip.initial_guess = cclip._compute_an_initial_guess(client_grad_list)
    result = cclip.robustify_global_model(avg_params, previous_global_w=None)
    print(f"result = {result}")


if __name__ == "__main__":
    test_defense()
    test_robustify_global_model()
