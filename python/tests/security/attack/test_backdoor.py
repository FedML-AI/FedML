from fedml.core.security.attack.backdoor_attack import BackdoorAttack
import logging
from utils import create_fake_model_list, create_fake_dataloader

logging.getLogger().setLevel(logging.INFO)


def test_get_malicious_client_idx():
    print(
        BackdoorAttack(backdoor_client_num=2, client_num=10)._get_malicious_client_idx(
            10
        )
    )


def test__backdoor():
    model_list = create_fake_model_list(10)
    dataset = create_fake_dataloader()
    attack = BackdoorAttack(
        backdoor_client_num=2, client_num=10, num_std=1.5, dataset=dataset
    )
    print(attack.attack_model(raw_client_grad_list=model_list))


if __name__ == "__main__":
    test_get_malicious_client_idx()
    test__backdoor()
