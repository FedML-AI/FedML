import torch
from torch.utils.data import TensorDataset
from fedml.data.cifar10.data_loader import load_partition_data_cifar10
from torch.utils.data import DataLoader
import pickle


class AttackDefenseDataLoader:
    @classmethod
    def load_cifar10_data(
        cls, client_num, batch_size, data_dir="../../../../../data/cifar10", partition_method="homo", partition_alpha=None
    ):
        """
        Load CIFAR-10 dataset and partition it among clients.

        Args:
            client_num (int): The number of clients to partition the dataset for.
            batch_size (int): The batch size for DataLoader objects.
            data_dir (str): The directory where the CIFAR-10 dataset is located.
            partition_method (str): The method for partitioning the dataset among clients.
            partition_alpha (float): The alpha parameter for partitioning (used when partition_method is "hetero").

        Returns:
            dict: A dictionary containing DataLoader objects for each client.
        """
        return load_partition_data_cifar10(
            "cifar10",
            data_dir=data_dir,
            partition_method=partition_method,
            partition_alpha=partition_alpha,
            client_number=client_num,
            batch_size=batch_size,
        )

    @classmethod
    def get_data_loader_from_data(cls, batch_size, X, Y, **kwargs):
        """
        Get a data loader created from a given set of data.

        Args:
            batch_size (int): Batch size of the DataLoader.
            X (numpy.ndarray): Data features.
            Y (numpy.ndarray): Data labels.
            **kwargs: Additional arguments for DataLoader.

        Returns:
            torch.utils.data.DataLoader: DataLoader object for the provided data.
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
        Load a DataLoader object from a file.

        Args:
            filename (str): The name of the file containing the DataLoader object.

        Returns:
            torch.utils.data.DataLoader: Loaded DataLoader object.
        """
        print("Loading data loader from file: {}".format(filename))

        with open(filename, "rb") as file:
            return pickle.load(file)
