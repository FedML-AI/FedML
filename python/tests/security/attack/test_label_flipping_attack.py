import argparse
import os
from os.path import expanduser

from fedml.core.security.attack.label_flipping_attack import LabelFlippingAttack
from fedml.core.security.common.attack_defense_data_loader import AttackDefenseDataLoader
from fedml.data.MNIST.data_loader import load_partition_data_mnist, download_mnist
from fedml.data.cifar100.data_loader import load_partition_data_cifar100

CLIENT_NUM = 1
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
    parser.add_argument("--data_cache_dir", type=str, default=os.path.join(expanduser("~"), "fedml_data"))
    parser.add_argument("--model", type=str, default='lr')

    args, unknown = parser.parse_known_args()
    return args


def test_attack_cifar10():
    args = add_args()
    dataset = AttackDefenseDataLoader.load_cifar10_data(client_num=CLIENT_NUM, batch_size=BATCH_SIZE)
    # replace class 3 with class 9; replace class 2 with class 1
    label_flipping_attack = LabelFlippingAttack(args)
    label_flipping_attack.poison_data(dataset[5][0])


def test_attack_cifar100():
    args = add_args()
    dataset = load_partition_data_cifar100(dataset='cifar100', data_dir=os.path.join(expanduser("~"), "fedml_data", "cifar100"), partition_method='homo', partition_alpha=0.5, client_number=CLIENT_NUM, batch_size=BATCH_SIZE)
    # replace class 3 with class 9; replace class 2 with class 1
    label_flipping_attack = LabelFlippingAttack(args)
    label_flipping_attack.poison_data(dataset[5][0])


def test_attack_mnist():
    args = add_args()
    download_mnist(args.data_cache_dir)
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_mnist(
        args,
        args.batch_size,
        train_path=os.path.join(args.data_cache_dir, "MNIST", "train"),
        test_path=os.path.join(args.data_cache_dir, "MNIST", "test"),
    )

    print(f"number of local datasets = {len(train_data_local_dict)}")
    # replace class 3 with class 9; replace class 2 with class 1
    label_flipping_attack = LabelFlippingAttack(args)
    label_flipping_attack.poison_data(train_data_local_dict[2])


if __name__ == "__main__":
    # test_attack_cifar10()
    # test_attack_mnist()
    test_attack_cifar100()
