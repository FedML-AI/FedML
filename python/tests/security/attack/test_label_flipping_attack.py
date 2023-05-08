import argparse

from fedml.core.security.attack.label_flipping_attack import LabelFlippingAttack
from fedml.core.security.common.attack_defense_data_loader import AttackDefenseDataLoader

CLIENT_NUM = 3
BATCH_SIZE = 32

def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file", "--cf", help="yaml configuration file", type=str, default="",
    )
    parser.add_argument("--original_class_list", type=list, default=[3, 2])
    parser.add_argument("--target_class_list", type=list, default=[9, 1])
    parser.add_argument("--client_num_per_round", type=int, default=CLIENT_NUM)
    parser.add_argument("--ratio_of_poisoned_client", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--comm_round", type=int, default=10)

    args, unknown = parser.parse_known_args()
    return args


def test_attack_cifar10():
    dataset = AttackDefenseDataLoader.load_cifar10_data(client_num=CLIENT_NUM, batch_size=BATCH_SIZE)
    # replace class 3 with class 9; replace class 2 with class 1
    label_flipping_attack = LabelFlippingAttack(add_args())
    label_flipping_attack.poison_data(dataset[5][0])


if __name__ == "__main__":
    test_attack_cifar10()
