from .globals import *
import random
import math


def partition(X, Y, method="uniform", n_clients=N_CLIENTS, **kwargs):
    if isinstance(method, str):
        if method == "uniform":
            return uniform_partition(X, Y, n_clients)
        else:
            raise Exception("Unimplemented.")
    elif callable(method):
        return method(X, Y, **kwargs)
    else:
        raise Exception("Unknown type.")


def uniform_partition(X, Y, n_clients):
    shuffle(X, Y)
    batch_size = math.ceil(len(X) / n_clients)
    start = 0
    result = dict()
    for client_idx in range(n_clients):
        end = start + batch_size if (start + batch_size) < len(X) else len(X)
        result[client_idx] = {"X": X[start:end], "Y": Y[start:end]}
        start = end
    return result


def shuffle(X, Y):
    temp = list(zip(X, Y))
    random.shuffle(temp)
    X, Y = zip(*temp)
