import random
import math
import sys
from .globals import *


# def partition(data_loader, method="uniform", n_clients=N_CLIENTS, partition_keys=PARTITION_KEYS, **kwargs):
#     keys = []
#     values = []
#     for key in partition_keys:
#         if key in data_loader:
#             keys.append(key)
#             values.append(data_loader[key])
#     validate_inputs(values)
#     result = None
#     if isinstance(method, str):
#         if method == "uniform":
#             result = uniform_partition(keys, values, n_clients)
#         else:
#             raise Exception("Unimplemented.")
#     elif callable(method):
#         assert "attributes" in data_loader
#         result = method(keys, values, data_loader["attributes"], **kwargs)
#     else:
#         raise Exception("Unknown type.")
#     for key, value in result.items():
#         data_loader[key] = value
#
#
#
# def validate_inputs(values):
#     assert len(values) != 0
#     length = None
#     for value in values:
#         if length is None:
#             length = len(value)
#         else:
#             assert length == len(value)
#
#
# def uniform_partition(keys, values, n_clients):
#     values = shuffle(values)
#     length = len(values[0])
#     batch_size = math.ceil(length / n_clients)
#     result = dict()
#     for i, key in enumerate(keys):
#         result[key] = dict()
#         start = 0
#         for client_idx in range(n_clients):
#             end = start + batch_size if (start + batch_size) < length else length
#             result[key][client_idx] = list(values[i][start:end])
#             start = end
#     return result
#
#
# def shuffle(values):
#     temp = list(zip(*values))
#     random.shuffle(temp)
#     values = zip(*temp)
#     return list(values)

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

