import torch
from torch.utils.data import TensorDataset
from fedml.data.cifar10.data_loader import load_partition_data_cifar10
from torch.utils.data import DataLoader
import pickle


class AttackDefenseDataLoader:
    @classmethod
    def load_cifar10_data(
        cls, client_num, batch_size, partition_method="homo", partition_alpha=None
    ):
        return load_partition_data_cifar10(
            "cifar10",
            data_dir="../../../../../data/cifar10",
            partition_method=partition_method,
            partition_alpha=partition_alpha,
            client_number=client_num,
            batch_size=batch_size,
        )

    @classmethod
    def get_data_loader_from_data(cls, batch_size, X, Y, **kwargs):
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

        if (
            "classification_problem" in kwargs
            and kwargs["classification_problem"] == False
        ):
            Y_torch = torch.from_numpy(Y).float()
        else:
            Y_torch = torch.from_numpy(Y).long()
        dataset = TensorDataset(X_torch, Y_torch)
        kwargs.pop("classification_problem", None)
        return DataLoader(dataset, batch_size=batch_size, **kwargs)

    @classmethod
    def load_data_loader_from_file(cls, filename):
        """
        Loads DataLoader object from a file if available.

        :param filename: string
        """
        print("Loading data loader from file: {}".format(filename))

        with open(filename, "rb") as file:
            return pickle.load(file)
