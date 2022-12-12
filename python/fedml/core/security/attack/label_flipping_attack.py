import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from ..common.utils import (
    get_malicious_client_id_list,
    replace_original_class_with_target_class,
    log_client_data_statistics,
)

"""
ref: Tolpegin, Vale, Truex,  "Data Poisoning Attacks Against Federated Learning Systems."  (2021).
attack @client, added by Yuhui, 07/08/2022
"""


class LabelFlippingAttack:
    def __init__(
        self, args):
        self.original_class_list = args.original_class_list
        self.target_class_list = args.target_class_list
        self.client_num = args.client_num
        self.attack_epoch = 0
        self.poisoned_client_num = args.poisoned_client_num
        self.batch_size = args.batch_size
        self.poisoned_client_list = []

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
            random_seed=self.attack_epoch, client_num=self.client_num, malicious_client_num=self.poisoned_client_num,
        )
        self.attack_epoch += 1
        poisoned_dataset = []
        for client_idx in range(self.client_num):
            if client_idx in self.poisoned_client_list:
                tmp_local_dataset_X = torch.Tensor([])
                tmp_local_dataset_Y = torch.Tensor([])
                for batch_idx, (data, target) in enumerate(train_data_local_dict[client_idx]):
                    tmp_local_dataset_X = torch.cat((tmp_local_dataset_X, data))
                    tmp_local_dataset_Y = torch.cat((tmp_local_dataset_Y, target))
                tmp_Y = replace_original_class_with_target_class(
                    data_labels=tmp_local_dataset_Y,
                    original_class_list=self.original_class_list,
                    target_class_list=self.target_class_list,
                )
                dataset = TensorDataset(tmp_local_dataset_X, tmp_Y)
                data_loader = DataLoader(dataset, batch_size=self.batch_size)
                poisoned_dataset.append(data_loader)
            else:
                poisoned_dataset.append(train_data_local_dict[client_idx])
        log_client_data_statistics(self.poisoned_client_list, poisoned_dataset)
        self.attack_epoch = self.attack_epoch + 1
        return poisoned_dataset
