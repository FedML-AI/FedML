import math
import random

from ..globals import *


def uniform_partition(train_index_list, test_index_list=None, n_clients=N_CLIENTS):
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
