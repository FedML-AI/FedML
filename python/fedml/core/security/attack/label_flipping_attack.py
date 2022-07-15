import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from fedml.core.security.common.utils import (
    get_malicious_client_id_list,
    replace_original_class_with_target_class,
)

"""
ref: Tolpegin, Vale, Truex,  "Data Poisoning Attacks Against Federated Learning Systems."  (2021).
attack @client, added by Yuhui, 07/08/2022
"""
def get_data_loader_from_data(batch_size, X, Y, **kwargs):
    """
    Get a data loader created from a given set of data.

    :param batch_size: batch size of data loader
    :type batch_size: int
    :param X: data features
    :type X: numpy.Array()
    :param Y: data labels
    :type Y: numpy.Array()
    :return: torch.utils.data.DataLoader
    """
    X_torch = torch.from_numpy(X).float()

    if "classification_problem" in kwargs and kwargs["classification_problem"] == False:
        Y_torch = torch.from_numpy(Y).float()
    else:
        Y_torch = torch.from_numpy(Y).long()
    dataset = TensorDataset(X_torch, Y_torch)

    kwargs.pop("classification_problem", None)

    return DataLoader(dataset, batch_size=batch_size, **kwargs)

def log_client_data_statistics(poisoned_client_ids, train_data_local_dict):
    """
    Logs all client data statistics.

    :param poisoned_client_ids: list of malicious clients
    :type poisoned_client_ids: list
    :param train_data_local_dict: distributed dataset
    :type train_data_local_dict: list(tuple)
    """
    for client_idx in range(len(train_data_local_dict)):
        if client_idx in poisoned_client_ids:
            targets_set = {}
            for _, (_, targets) in enumerate(train_data_local_dict[client_idx]):
                for target in targets.numpy():
                    if target not in targets_set.keys():
                        targets_set[target] = 1
                    else:
                        targets_set[target] += 1
            print("Client #{} has data distribution:".format(client_idx))
            for item in targets_set.items():
                print("target:{} num:{}".format(item[0], item[1]))


class LabelFlippingAttack:
    def __init__(
        self,
        original_class,
        target_class,
        client_num,
        poisoned_client_num,
        batch_size,
    ):
        self.original_class = (original_class,)
        self.target_class = (target_class,)
        self.client_num = client_num
        self.attack_epoch = 0
        self.poisoned_client_num = poisoned_client_num
        self.batch_size = batch_size
        self.poisoned_client_list = []

    def attack_on_data_labels(self, dataset):
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
        poisoned_dataset = []
        for client_idx in range(self.client_num):
            if client_idx in self.poisoned_client_list:
                tmp_local_dataset_X = torch.Tensor([])
                tmp_local_dataset_Y = torch.Tensor([])
                for batch_idx, (data, target) in enumerate(
                    train_data_local_dict[client_idx]
                ):
                    tmp_local_dataset_X = torch.cat((tmp_local_dataset_X, data))
                    tmp_local_dataset_Y = torch.cat((tmp_local_dataset_Y, target))
                tmp_Y = replace_original_class_with_target_class(
                    data_labels=tmp_local_dataset_Y, original_class=self.original_class, target_class=self.target_class
                )
                dataset = TensorDataset(tmp_local_dataset_X, tmp_Y)
                data_loader = DataLoader(dataset, batch_size=self.batch_size)
                poisoned_dataset.append(data_loader)
            else:
                poisoned_dataset.append(train_data_local_dict[client_idx])
        log_client_data_statistics(self.poisoned_client_list, poisoned_dataset)
        return poisoned_dataset
