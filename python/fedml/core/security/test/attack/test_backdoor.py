# import os, sys
# __file__ = '/Users/kai/Documents/FedML/python/'
# sys.path.append(__file__)
from fedml.core.security.attack.backdoor_attack import BackdoorAttack
from fedml.core.security.test.utils import (
    create_fake_model_list,
    create_fake_dataloader,
)

import logging

logging.getLogger().setLevel(logging.INFO)


def test_get_malicious_client_idx():
    print(
        BackdoorAttack(backdoor_client_num=2, client_num=10)._get_malicious_client_idx(
            10
        )
    )


def test__backdoor():
    local_w = dict(create_fake_model_list(10))
    dataset = create_fake_dataloader()
    attack = BackdoorAttack(
        backdoor_client_num=2, client_num=10, num_std=1.5, dataset=dataset
    )
    logging.info(attack.attack_model(local_w, global_w=None))


if __name__ == "__main__":
    test_get_malicious_client_idx()
    test__backdoor()
