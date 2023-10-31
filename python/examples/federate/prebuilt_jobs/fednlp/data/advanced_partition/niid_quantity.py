import h5py
import argparse
import json
import numpy as np


def partition_class_samples_with_dirichlet_distribution(
    N, alpha, client_num, idx_batch, idx_k
):
    """
    params
    ------------------------------------
    N : int  total length of the dataset
    alpha : int  similarity of each client, the larger the alpha the similar data for each client
    client_num : int number of clients
    idx_batch: 2d list shape(client_num, ?), this is the list of index for each client
    idx_k : 1d list  list of index of the dataset
    ------------------------------------

    return
    ------------------------------------
    idx_batch : 2d list shape(client_num, ?) list of index for each client
    min_size : minimum size of all the clients' sample
    ------------------------------------
    """
    # first shuffle the index
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the new batch list for each client
    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]

    return idx_batch


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client_number",
        type=int,
        default="100",
        metavar="CN",
        help="client number for lda partition",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default="data/data_files/20news_data.h5",
        metavar="DF",
        help="data pickle file path",
    )

    parser.add_argument(
        "--partition_file",
        type=str,
        default="data/partition_files/20news_partition.h5",
        metavar="PF",
        help="partition pickle file path",
    )

    parser.add_argument("--task_type", type=str, metavar="TT", help="task type")

    parser.add_argument(
        "--kmeans_num", type=int, metavar="KN", help="number of k-means cluster"
    )

    parser.add_argument(
        "--beta", type=float, metavar="B", help="beta value for quantity skew"
    )

    args = parser.parse_args()
    print("start reading data")
    client_num = args.client_number
    beta = args.beta  # need adjustment for each dataset

    data = h5py.File(args.data_file, "r")
    attributes = json.loads(data["attributes"][()])
    total_index_len = len(attributes["index_list"])
    train_index_list = []
    test_index_list = []

    label_list = attributes["index_list"]
    if "train_index_list" in attributes:
        test_index_list = attributes["test_index_list"]
        train_index_list = attributes["train_index_list"]
    else:
        train_length = int(total_index_len * 0.9)
        train_index_list = label_list[0:train_length]
        test_index_list = label_list[train_length:]

    min_size_test = 0
    min_size_train = 0

    print("start dirichlet distribution")
    while min_size_test < 1 or min_size_train < 1:
        partition_result_train = [[] for _ in range(client_num)]
        partition_result_test = [[] for _ in range(client_num)]
        train_n = len(train_index_list)
        test_n = len(test_index_list)
        partition_result_train = partition_class_samples_with_dirichlet_distribution(
            train_n, beta, client_num, partition_result_train, train_index_list
        )
        partition_result_test = partition_class_samples_with_dirichlet_distribution(
            test_n, beta, client_num, partition_result_test, test_index_list
        )
    print("minsize of the train data", min([len(i) for i in partition_result_train]))
    print("minsize of the test data", min([len(i) for i in partition_result_test]))
    data.close()

    print("store data in h5 data")
    partition = h5py.File(args.partition_file, "a")

    if (
        "/niid_quantity_clients_%d_beta=%.1f" % (args.client_number, args.beta)
        in partition
    ):
        del partition[
            "/niid_quantity_clients_%d_beta=%.1f" % (args.client_number, args.beta)
        ]
    if (
        "/niid_quantity_clients=%d_beta=%.1f" % (args.client_number, args.beta)
        in partition
    ):
        del partition[
            "/niid_quantity=clients_%d_beta=%.1f" % (args.client_number, args.beta)
        ]

    partition[
        "/niid_quantity_clients=%d_beta=%.1f" % (args.client_number, args.beta)
        + "/n_clients"
    ] = client_num
    partition[
        "/niid_quantity_clients=%d_beta=%.1f" % (args.client_number, args.beta)
        + "/beta"
    ] = beta
    for partition_id in range(client_num):
        train = partition_result_train[partition_id]
        test = partition_result_test[partition_id]
        train_path = (
            "/niid_quantity_clients=%d_beta=%.1f" % (args.client_number, args.beta)
            + "/partition_data/"
            + str(partition_id)
            + "/train/"
        )
        test_path = (
            "/niid_quantity_clients=%d_beta=%.1f" % (args.client_number, args.beta)
            + "/partition_data/"
            + str(partition_id)
            + "/test/"
        )
        partition[train_path] = train
        partition[test_path] = test
    partition.close()


main()
