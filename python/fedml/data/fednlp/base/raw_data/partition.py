import math
import random

from ..globals import *


def uniform_partition(train_index_list, test_index_list=None, n_clients=N_CLIENTS):
    """Uniformly partition data indices into multiple clients.

    This function partitions a list of training data indices into 'n_clients' subsets,
    ensuring a roughly equal distribution of data among clients. Optionally, it can also
    partition a list of test data indices in a similar manner.

    Args:
        train_index_list (list): List of training data indices.
        test_index_list (list, optional): List of test data indices. Default is None.
        n_clients (int): Number of clients to partition the data for.

    Returns:
        dict: A dictionary containing the data partition information.
            - 'n_clients': Number of clients.
            - 'partition_data': A dictionary where each key represents a client ID (0 to n_clients-1),
                and the value is another dictionary containing the partitioned data for that client.
                For each client:
                - 'train': List of training data indices.
                - 'test': List of test data indices (if 'test_index_list' is provided).

    """
    partition_dict = dict()
    partition_dict["n_clients"] = n_clients
    partition_dict["partition_data"] = dict()
    train_index_list = train_index_list.copy()
    random.shuffle(train_index_list)
    train_batch_size = math.ceil(len(train_index_list) / n_clients)

    test_batch_size = None

    if test_index_list is not None:
        test_index_list = test_index_list.copy()
        random.shuffle(test_index_list)
        test_batch_size = math.ceil(len(test_index_list) / n_clients)
    for i in range(n_clients):
        train_start = i * train_batch_size
        partition_dict["partition_data"][i] = dict()
        train_set = train_index_list[train_start : train_start + train_batch_size]
        if test_index_list is None:
            random.shuffle(train_set)
            train_num = int(len(train_set) * 0.8)
            partition_dict["partition_data"][i]["train"] = train_set[:train_num]
            partition_dict["partition_data"][i]["test"] = train_set[train_num:]
        else:
            test_start = i * test_batch_size
            partition_dict["partition_data"][i]["train"] = train_set
            partition_dict["partition_data"][i]["test"] = test_index_list[
                test_start : test_start + test_batch_size
            ]

    return partition_dict
