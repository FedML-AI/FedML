import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

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


def apply_class_label_replacement(X, Y, replacement_method):
    """
    Replace class labels using the replacement method

    :param X: data features
    :type X: numpy.Array()
    :param Y: data labels
    :type Y: numpy.Array()
    :param replacement_method: Method to update targets
    :type replacement_method: method
    """
    return X, replacement_method(Y, set(Y))


def log_client_data_statistics(poisoned_client_ids, train_data_local_dict):
    """
    Logs all client data statistics.

    :param label_class_set: set of class labels
    :type label_class_set: list
    :param distributed_dataset: distributed dataset
    :type distributed_dataset: list(tuple)
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
        replacement_method,
        client_num,
        worker_num,
        attack_epoch,
        attack_client_num,
        batch_size,
    ):
        self.replacement_method = replacement_method
        self.worker_num = worker_num
        self.client_num_in_total = client_num
        self.attack_epoch = attack_epoch
        self.attack_client_num = attack_client_num
        self.batch_size = batch_size

    def attack(self, local_w, global_w, dataset, refs=None):
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

        poison_loacal_train_data = self.poison_data(
            train_data_local_dict, self.client_num_in_total, self.attack_client_num
        )
        return poison_loacal_train_data

    def poison_data(
        self, train_data_local_dict, client_num_in_total, attack_client_num
    ):
        """
        Poison worker data

        :param distributed_dataset: Distributed dataset
        :type distributed_dataset: list(tuple)
        :param num_workers: Number of workers overall
        :type num_workers: int
        :param poisoned_worker_ids: IDs poisoned workers
        :type poisoned_worker_ids: list(int)
        :param replacement_method: Replacement methods to use to replace
        :type replacement_method: list(method)
        """
        poisoned_client_ids = self._client_sampling(
            self.attack_epoch, client_num_in_total, self.attack_client_num
        )

        # TODO: Add support for multiple replacement methods?
        poisoned_dataset = []

        # class_labels = list(set(distributed_dataset[0][1]))

        # print("Poisoning data for workers: {}".format(str(poisoned_worker_ids)))

        for client_idx in range(client_num_in_total):
            if client_idx in poisoned_client_ids:
                tmp_local_dataset_X = torch.Tensor([])
                tmp_local_dataset_Y = torch.Tensor([])

                for batch_idx, (data, target) in enumerate(
                    train_data_local_dict[client_idx]
                ):
                    tmp_local_dataset_X = torch.cat((tmp_local_dataset_X, data))
                    tmp_local_dataset_Y = torch.cat((tmp_local_dataset_Y, target))

                tmp_X, tmp_Y = apply_class_label_replacement(
                    tmp_local_dataset_X, tmp_local_dataset_Y, self.replacement_method
                )

                dataset = TensorDataset(tmp_X, tmp_Y)

                data_loader = DataLoader(dataset, batch_size=self.batch_size)
                poisoned_dataset.append(data_loader)
            else:
                poisoned_dataset.append(train_data_local_dict[client_idx])

        log_client_data_statistics(poisoned_client_ids, poisoned_dataset)

        return poisoned_dataset

    def get_poisoned_client_id_list(
        self, round_idx, client_num_in_total, poisoned_client_num
    ):
        if client_num_in_total == poisoned_client_num:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(poisoned_client_num, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        print("client_indexes = %s" % str(client_indexes))
        return client_indexes
