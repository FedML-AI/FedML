import argparse
from fedml.core.security.attack.byzantine_attack import ByzantineAttack
from utils import create_fake_model_list


def add_args(byzantine_client_num, attack_mode):
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    # default arguments
    parser.add_argument(
        "--byzantine_client_num", type=int, default=byzantine_client_num
    )
    parser.add_argument("--attack_mode", type=str, default=attack_mode)

    args, unknown = parser.parse_known_args()
    return args


def test_get_malicious_client_idx():
    config = add_args(byzantine_client_num=2, attack_mode="zero")
    print(ByzantineAttack(config)._get_malicious_client_idx(10))
    config = add_args(byzantine_client_num=10, attack_mode="zero")
    print(ByzantineAttack(config)._get_malicious_client_idx(10))


def test__attack_zero_mode():
    local_w = create_fake_model_list(10)
    attack = ByzantineAttack(add_args(byzantine_client_num=2, attack_mode="zero"))
    print(attack.attack_model(local_w))


def test__attack_random_mode():
    local_w = create_fake_model_list(10)
    attack = ByzantineAttack(add_args(byzantine_client_num=2, attack_mode="random"))
    print(attack.attack_model(local_w))


def test__attack_flip_mode():
    local_w = create_fake_model_list(10)
    attack = ByzantineAttack(add_args(byzantine_client_num=2, attack_mode="flip"))
    print(attack.attack_model(local_w, extra_auxiliary_info=local_w[0][1]))
