import numpy as np


def client_sampling(round_idx, client_num_in_total, client_num_per_round):
    if client_num_in_total == client_num_per_round:
        client_indexes = [client_index for client_index in range(client_num_in_total)]
    else:
        num_clients = min(client_num_per_round, client_num_in_total)
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
    print("client_indexes = %s" % str(client_indexes))
    return client_indexes
