import random
import math
import sys
from .globals import *


def uniform_partition(train_data_list, test_data_list=None, n_clients=N_CLIENTS):
    partition_dict = dict()
    partition_dict["n_clients"] = n_clients
    partition_dict["partition_data"] = dict()
    train_batch_size = math.ceil(len(train_data_list[0]) / n_clients)
    train_index_list = [i for i in range(len(train_data_list[0]))]

    test_batch_size = None
    test_index_list = None
    if test_data_list is not None:
        test_batch_size = math.ceil(len(test_data_list[0]) / n_clients)
        test_index_list = [i + len(train_data_list[0]) for i in range(len(test_data_list[0]))]
    for i in range(n_clients):
        train_start = i * train_batch_size
        partition_dict["partition_data"][i] = dict()
        train_set = train_index_list[train_start: train_start+train_batch_size]
        if test_data_list is None:
            random.shuffle(train_set)
            train_num = int(len(train_set) * 0.8)
            partition_dict["partition_data"][i]["train"] = train_set[:train_num]
            partition_dict["partition_data"][i]["test"] = train_set[train_num:]
        else:
            test_start = i * test_batch_size
            partition_dict["partition_data"][i]["train"] = train_set
            partition_dict["partition_data"][i]["test"] = test_index_list[test_start:test_start+test_batch_size]

    return partition_dict

