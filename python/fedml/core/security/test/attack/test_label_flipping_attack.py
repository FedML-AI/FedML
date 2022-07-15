from fedml.core.security.attack.label_flipping_attack import LabelFlippingAttack
from fedml.core.security.common.attack_defense_data_loader import (
    AttackDefenseDataLoader,
)
from fedml.data.MNIST.data_loader import load_partition_data_mnist


def test_attack_cifar10():
    client_num = 10
    attack_client_num = 5
    batch_size = 32
    dataset = AttackDefenseDataLoader.load_cifar10_data(
        client_num=client_num, batch_size=batch_size
    )
    # replace class 3 with class 9; replace class 2 with class 1
    label_flipping_attack = LabelFlippingAttack(
        original_class_list=[3, 2],
        target_class_list=[9, 1],
        client_num=client_num,
        poisoned_client_num=attack_client_num,
        batch_size=batch_size,
    )
    label_flipping_attack.attack_on_data_labels(dataset)


def test_attack_mnist():
    client_num = 10
    attack_client_num = 5
    batch_size = 256
    dataset = load_partition_data_mnist(
        None,
        batch_size,
        train_path="../../../../../data/mnist/MNIST/train",
        test_path="../../../../../data/mnist/MNIST/test",
    )
    label_flipping_attack = LabelFlippingAttack(
        [0], [6], client_num, attack_client_num, batch_size
    )
    label_flipping_attack.attack_on_data_labels(dataset)


if __name__ == "__main__":
    test_attack_cifar10()
