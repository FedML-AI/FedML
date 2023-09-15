import numpy as np


def client_sampling(round_idx, client_num_in_total, client_num_per_round):
    """
    Sample a subset of clients for federated learning.

    Args:
        round_idx (int): The index of the current federated learning round.
        client_num_in_total (int): The total number of available clients.
        client_num_per_round (int): The number of clients to select for the current round.

    Returns:
        list: A list of selected client indexes for the current round.
    """
    if client_num_in_total == client_num_per_round:
        client_indexes = [client_index for client_index in range(client_num_in_total)]
    else:
        num_clients = min(client_num_per_round, client_num_in_total)
        np.random.seed(round_idx)  # Make sure for each comparison, we select the same clients each round.
        client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
    print("client_indexes = %s" % str(client_indexes))
    return client_indexes
