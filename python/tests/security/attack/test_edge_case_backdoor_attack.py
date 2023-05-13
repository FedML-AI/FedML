import os.path
from os.path import expanduser

import torch
from torch.utils.data import TensorDataset
from fedml.core.security.attack.edge_case_backdoor_attack import EdgeCaseBackdoorAttack
from fedml.core.security.common.attack_defense_data_loader import AttackDefenseDataLoader


def test_attack_cifar10():
    client_num = 10
    attack_client_num = 5
    batch_size = 32
    data_dir = os.path.join(expanduser("~"), "fedml_data", "cifar10")
    dataset = AttackDefenseDataLoader.load_cifar10_data(data_dir=data_dir, client_num=client_num, batch_size=batch_size)
    backdoor_data = torch.rand(784, 3, 32, 32)
    backdoor_target = torch.randint(10, (784,))
    backdoor_dataset = TensorDataset(backdoor_data, backdoor_target)
    edge_case_backdoor_attack = EdgeCaseBackdoorAttack(
        client_num=client_num,
        poisoned_client_num=attack_client_num,
        backdoor_sample_percentage=0.1,
        backdoor_dataset=backdoor_dataset,
        batch_size=batch_size,
    )
    edge_case_backdoor_attack.poison_data(dataset)


if __name__ == "__main__":
    test_attack_cifar10()
