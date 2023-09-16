import json
import logging

import numpy as np
import torch
import torch.utils.data as data

DEFAULT_BATCH_SIZE = 16
train_file_path = "../../../data/synthetic_1_1/train/mytrain.json"
test_file_path = "../../../data/synthetic_1_1/test/mytest.json"

_USERS = "users"
_USER_DATA = "user_data"


def load_partition_data_federated_synthetic_1_1(
    train_file_path, test_file_path, batch_size=DEFAULT_BATCH_SIZE
):
    """
    Load federated synthetic data for training and testing.

    Args:
        train_file_path (str): Path to the training data JSON file.
        test_file_path (str): Path to the testing data JSON file.
        batch_size (int): Batch size for data loaders.

    Returns:
        tuple: A tuple containing client and data-related information:
            - client_num (int): Number of clients.
            - train_data_num (int): Number of samples in the global training dataset.
            - test_data_num (int): Number of samples in the global testing dataset.
            - train_data_global (torch.utils.data.DataLoader): DataLoader for the global training dataset.
            - test_data_global (torch.utils.data.DataLoader): DataLoader for the global testing dataset.
            - data_local_num_dict (dict): Dictionary containing the number of samples for each client.
            - train_data_local_dict (dict): Dictionary of DataLoader objects for local training data.
            - test_data_local_dict (dict): Dictionary of DataLoader objects for local testing data.
            - output_dim (int): Dimension of the output (e.g., number of classes).
    """
    logging.info("load_partition_data_federated_synthetic_1_1 START")

    with open(train_file_path, "r") as train_f, open(test_file_path, "r") as test_f:
        train_data = json.load(train_f)
        test_data = json.load(test_f)

        client_ids_train = train_data[_USERS]
        client_ids_test = test_data[_USERS]
        client_num = len(train_data[_USERS])

        full_x_train = torch.from_numpy(np.asarray([])).float()
        full_y_train = torch.from_numpy(np.asarray([])).long()
        full_x_test = torch.from_numpy(np.asarray([])).float()
        full_y_test = torch.from_numpy(np.asarray([])).long()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for i in range(len(client_ids_train)):
            train_ds = data.TensorDataset(
                torch.tensor(train_data[_USER_DATA][client_ids_train[i]]["x"]),
                torch.tensor(
                    train_data[_USER_DATA][client_ids_train[i]]["y"], dtype=torch.int64
                ),
            )
            test_ds = data.TensorDataset(
                torch.tensor(train_data[_USER_DATA][client_ids_test[i]]["x"]),
                torch.tensor(
                    train_data[_USER_DATA][client_ids_test[i]]["y"], dtype=torch.int64
                ),
            )
            train_dl = data.DataLoader(
                dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=False
            )
            test_dl = data.DataLoader(
                dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=False
            )
            train_data_local_dict[i] = train_dl
            test_data_local_dict[i] = test_dl

            full_x_train = torch.cat(
                (
                    full_x_train,
                    torch.tensor(train_data[_USER_DATA][client_ids_train[i]]["x"]),
                ),
                0,
            )
            full_y_train = torch.cat(
                (
                    full_y_train,
                    torch.tensor(
                        train_data[_USER_DATA][client_ids_train[i]]["y"],
                        dtype=torch.int64,
                    ),
                ),
                0,
            )
            full_x_test = torch.cat(
                (
                    full_x_test,
                    torch.tensor(test_data[_USER_DATA][client_ids_test[i]]["x"]),
                ),
                0,
            )
            full_y_test = torch.cat(
                (
                    full_y_test,
                    torch.tensor(
                        test_data[_USER_DATA][client_ids_test[i]]["y"],
                        dtype=torch.int64,
                    ),
                ),
                0,
            )

        train_ds = data.TensorDataset(full_x_train, full_y_train)
        test_ds = data.TensorDataset(full_x_test, full_y_test)
        train_data_global = data.DataLoader(
            dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )
        test_data_global = data.DataLoader(
            dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=False
        )
        train_data_num = len(train_data_global.dataset)
        test_data_num = len(test_data_global.dataset)
        data_local_num_dict = {
            i: len(train_data_local_dict[i].dataset) for i in train_data_local_dict
        }
        output_dim = 10

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        output_dim,
    )


def test_data_loader(train_file_path):
    """
    Test the data loader function by comparing the number of samples with the original data.

    Args:
        train_file_path (str): Path to the training data JSON file.
    """
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        output_dim,
    ) = load_partition_data_federated_synthetic_1_1(train_file_path, train_file_path)
    
    with open(train_file_path, "r") as f:
        train_data = json.load(f)
    assert train_data["num_samples"] == list(data_local_num_dict.values())


if __name__ == "__main__":
    test_data_loader()
