import argparse
from fedml.core.security.attack.model_replacement_backdoor_attack import ModelReplacementBackdoorAttack
from utils import create_fake_model_list


def add_default_setting2():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    # default arguments
    parser.add_argument("--scale_factor_S", type=float, default=5.0)
    args, unknown = parser.parse_known_args()
    return args


def test_attack():
    raw_client_grad_list = create_fake_model_list(10)
    print(f"before: {raw_client_grad_list}")
    attack = ModelReplacementBackdoorAttack(args=None)
    print(f"after:{attack.attack_model(raw_client_grad_list, extra_auxiliary_info=raw_client_grad_list[0][1])}")


def test_attack_with_scale_factor_bound():
    raw_client_grad_list = create_fake_model_list(10)
    print(f"before: {raw_client_grad_list}")
    attack = ModelReplacementBackdoorAttack(args=None)
    print(f"after:{attack.attack_model(raw_client_grad_list, extra_auxiliary_info=raw_client_grad_list[0][1])}")
