import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np

from ..common.utils import (
    get_malicious_client_id_list,
    replace_original_class_with_target_class,
    log_client_data_statistics,
)

"""
attack @client, added by Xiaoyang, 08/03/2022
"Attack of the Tails: Yes, You Really Can Backdoor Federated Learning"
https://arxiv.org/pdf/2007.05084.pdf
"""


class EdgeCaseBackdoorAttack:
    def __init__(
        self,
        client_num,
        poisoned_client_num,
        backdoor_sample_percentage,
        backdoor_dataset,
        batch_size,
    ):
        self.client_num = client_num
        self.attack_epoch = 0
        self.poisoned_client_num = poisoned_client_num
        self.backdoor_sample_percentage = backdoor_sample_percentage
        self.backdoor_dataset = backdoor_dataset
        self.poisoned_client_list = []
        self.batch_size = batch_size

    def poison_data(self, dataset):
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
        self.poisoned_client_list = get_malicious_client_id_list(
            random_seed=self.attack_epoch,
            client_num=self.client_num,
            malicious_client_num=self.poisoned_client_num,
        )
        self.attack_epoch += 1
        backdoored_dataset = []
        for client_idx in range(self.client_num):
            if client_idx in self.poisoned_client_list:
                tmp_local_dataset_X = torch.Tensor([])
                tmp_local_dataset_Y = torch.Tensor([])
                for batch_idx, (data, target) in enumerate(
                    train_data_local_dict[client_idx]
                ):
                    backdoor_sample_num = int(
                        self.batch_size * self.backdoor_sample_percentage
                    )
                    backdoor_sample_indices = np.random.choice(
                        len(list(self.backdoor_dataset)),
                        backdoor_sample_num,
                        replace=False,
                    )
                    backdoor_data, backdoor_target = self.backdoor_dataset[
                        backdoor_sample_indices
                    ]
                    # insert backdoor samples
                    data = torch.cat(
                        (data[0 : self.batch_size - backdoor_sample_num], backdoor_data)
                    )
                    target = torch.cat(
                        (
                            target[0 : self.batch_size - backdoor_sample_num],
                            backdoor_target,
                        )
                    )
                    tmp_local_dataset_X = torch.cat((tmp_local_dataset_X, data))
                    tmp_local_dataset_Y = torch.cat((tmp_local_dataset_Y, target))
                dataset = TensorDataset(tmp_local_dataset_X, tmp_local_dataset_Y)
                data_loader = DataLoader(dataset, batch_size=self.batch_size)
                backdoored_dataset.append(data_loader)
            else:
                backdoored_dataset.append(train_data_local_dict[client_idx])
        log_client_data_statistics(self.poisoned_client_list, backdoored_dataset)
        return backdoored_dataset
