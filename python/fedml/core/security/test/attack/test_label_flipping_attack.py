# from fedml.core.security.attack.byzantine_attack import get_malicious_client_idx, ByzantineAttack
from fedml.core.security.test.utils import create_fake_model_list

from fedml.core.security.attack.label_flipping_attack import LabelFlippingAttack
from fedml.core.security.attack.class_flipping_methods import *
from fedml.data.MNIST.data_loader import load_partition_data_mnist
from fedml.data.cifar10.data_loader import load_partition_data_cifar10, load_cifar10_data
import fedml
import os
import numpy as np
import pickle


def load_data_loader_from_file(filename):
    """
    Loads DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param filename: string
    """
    print("Loading data loader from file: {}".format(filename))

    with open(filename, "rb") as f:
        return load_saved_data_loader(f)


def load_saved_data_loader(file_obj):
    return pickle.load(file_obj)


def test_attack_cifar10():
    client_num = 10
    worker_num = 4
    attack_epoch = 1
    attack_client_num = 5
    batch_size = 32
    # init FedML framework
    # args = fedml.init()

    dataset = load_partition_data_cifar10(
        'cifar10',
        data_dir='../../../../../data/cifar10',
        partition_method='homo',
        partition_alpha=None,
        client_number=client_num,
        batch_size=batch_size,
    )

    label_flipping_attack = LabelFlippingAttack(replace_3_with_9, client_num, worker_num, attack_epoch,
                                                attack_client_num, batch_size)
    label_flipping_attack.attack(None, None, dataset)


def test_attack_mnist():
    client_num = 10
    worker_num = 4
    attack_epoch = 1
    attack_client_num = 5
    batch_size = 256
    # init FedML framework
    # args = fedml.init()

    dataset = load_partition_data_mnist(
        None,
        batch_size,
        # train_path="../../../../../data/mnist/train",
        train_path="../../../../../data/mnist/MNIST/train",
        test_path="../../../../../data/mnist/MNIST/test",
    )
    label_flipping_attack = LabelFlippingAttack(replace_0_with_6, client_num, worker_num, attack_epoch,
                                                attack_client_num)
    label_flipping_attack.attack(None, None, dataset, None)


if __name__ == '__main__':
    # print(os.curdir)
    test_attack_cifar10()
    # test_get_malicious_client_idx()

    # test__attack_test()
