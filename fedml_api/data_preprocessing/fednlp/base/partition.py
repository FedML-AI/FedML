import random
import math
import sys
from .globals import *


def partition(data_loader, method="uniform", n_clients=N_CLIENTS, partition_keys=PARTITION_KEYS, **kwargs):
    keys = []
    values = []
    for key in partition_keys:
        if key in data_loader:
            keys.append(key)
            values.append(data_loader[key])
    validate_inputs(values)
    result = None
    if isinstance(method, str):
        if method == "uniform":
            result = uniform_partition(keys, values, n_clients)
        else:
            raise Exception("Unimplemented.")
    elif callable(method):
        assert "attributes" in data_loader
        result = method(keys, values, data_loader["attributes"], **kwargs)
    else:
        raise Exception("Unknown type.")
    for key, value in result.items():
        data_loader[key] = value



def validate_inputs(values):
    assert len(values) != 0
    length = None
    for value in values:
        if length is None:
            length = len(value)
        else:
            assert length == len(value)


def uniform_partition(keys, values, n_clients):
    values = shuffle(values)
    length = len(values[0])
    batch_size = math.ceil(length / n_clients)
    result = dict()
    for i, key in enumerate(keys):
        result[key] = dict()
        start = 0
        for client_idx in range(n_clients):
            end = start + batch_size if (start + batch_size) < length else length
            result[key][client_idx] = list(values[i][start:end])
            start = end
    return result


def shuffle(values):
    temp = list(zip(*values))
    random.shuffle(temp)
    values = zip(*temp)
    return list(values)


