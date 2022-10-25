import argparse

from fedml.core.security.attack.label_flipping_attack import LabelFlippingAttack
from fedml.core.security.common.attack_defense_data_loader import AttackDefenseDataLoader


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file", "--cf", help="yaml configuration file", type=str, default="",
    )
    parser.add_argument("--original_class_list", type=list, default=[3, 2])
    parser.add_argument("--target_class_list", type=list, default=[9, 1])
    parser.add_argument("--client_num", type=int, default=10)
    parser.add_argument("--poisoned_client_num", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)

    args, unknown = parser.parse_known_args()
    return args


def test_attack_cifar10():
    client_num = 10
    attack_client_num = 5
    batch_size = 32
    dataset = AttackDefenseDataLoader.load_cifar10_data(client_num=client_num, batch_size=batch_size)
    # replace class 3 with class 9; replace class 2 with class 1
    label_flipping_attack = LabelFlippingAttack(add_args())
    label_flipping_attack.poison_data(dataset)


if __name__ == "__main__":
    test_attack_cifar10()
