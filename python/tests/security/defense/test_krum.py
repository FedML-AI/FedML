import argparse
from fedml.core.security.defense.krum_defense import KrumDefense
from utils import create_fake_model_list


def add_args_krum(isMultiKrum):
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file", "--cf", help="yaml configuration file", type=str, default="",
    )
    parser.add_argument("--byzantine_client_num", type=int, default=1)
    if isMultiKrum:
        parser.add_argument("--krum_param_m", type=bool, default=2)
    args, unknown = parser.parse_known_args()
    return args


def test_defense_Krum():
    model_list = create_fake_model_list(20)
    print(f"test krum")
    krum = KrumDefense(add_args_krum(isMultiKrum=False))
    filtered_model_list = krum.defend_before_aggregation(model_list)
    print(f"filtered_model_list={filtered_model_list}")


def test_defense_multi_Krum():
    model_list = create_fake_model_list(20)
    print(f"test multi-krum")
    krum = KrumDefense(add_args_krum(isMultiKrum=True))
    filtered_model_list = krum.defend_before_aggregation(model_list)
    print(f"filtered_model_list={filtered_model_list}")


if __name__ == "__main__":
    test_defense_Krum()
    test_defense_multi_Krum()
