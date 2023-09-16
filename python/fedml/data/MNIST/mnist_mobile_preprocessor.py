import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


def add_args(parser):
    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=3,
        metavar="NN",
        help="number of workers",
    )
    parser.add_argument(
        "--comm_round",
        type=int,
        default=10,
        help="how many round of communications we should use",
    )
    args = parser.parse_args()
    return args


def read_data(train_data_dir, test_data_dir):
    """
    Parse data from train and test data directories.
    
    Assumes:
    - Data in the input directories are .json files with keys 'users' and 'user_data'.
    - The set of train set users is the same as the set of test set users.

    Args:
        train_data_dir (str): Path to the directory containing train data.
        test_data_dir (str): Path to the directory containing test data.

    Returns:
        clients (list): List of client ids.
        train_num_samples (list): List of the number of samples for each client in the training data.
        test_num_samples (list): List of the number of samples for each client in the test data.
        train_data (dict): Dictionary of training data.
        test_data (dict): Dictionary of test data.
        client_list (list): List of client arguments.
    """
    clients = []
    train_num_samples = []
    test_num_samples = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    # print(train_files)
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        train_num_samples.extend(cdata["num_samples"])
        train_data.update(cdata["user_data"])
        # print(cdata['user_data'])
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        test_num_samples.extend(cdata["num_samples"])
        test_data.update(cdata["user_data"])

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    main_args = add_args(parser)

    class Args:
        def __init__(self, client_id, client_num_per_round, comm_round):
            self.client_num_per_round = client_num_per_round
            self.comm_round = comm_round
            self.client_id = client_id
            self.client_sample_list = []

    client_list = []
    for client_number in range(main_args.client_num_per_round):
        client_list.append(
            Args(client_number, main_args.client_num_per_round, main_args.comm_round)
        )
    return (
        clients,
        train_num_samples,
        test_num_samples,
        train_data,
        test_data,
        client_list,
    )


def client_sampling(round_idx, client_num_in_total, client_num_per_round):
    """
    Randomly select clients for federated learning.

    Args:
        round_idx (int): Index of the current federated learning round.
        client_num_in_total (int): Total number of clients available.
        client_num_per_round (int): Number of clients to select for the current round.

    Returns:
        client_indexes (list): List of selected client indexes for the current round.
    """

    if client_num_in_total == client_num_per_round:
        client_indexes = [client_index for client_index in range(client_num_in_total)]
    else:
        num_clients = min(client_num_per_round, client_num_in_total)
        np.random.seed(
            round_idx
        )  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(
            range(client_num_in_total), num_clients, replace=False
        )
    print("client_indexes = %s" % str(client_indexes))
    return client_indexes


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    main_args = add_args(parser)
    train_path = "../../FedML/data/MNIST/train"
    test_path = "../../FedML/data/MNIST/test"
    new_train = {}
    new_test = {}

    (
        users,
        train_num_samples,
        test_num_samples,
        train_data,
        test_data,
        client_list,
    ) = read_data(train_path, test_path)

    for round_idx in range(client_list[0].comm_round):
        sample_list = client_sampling(round_idx, 1000, main_args.client_num_per_round)
        for worker in client_list:
            worker.client_sample_list.append(sample_list[worker.client_id])
    os.mkdir("MNIST_mobile_zip")
    for worker in client_list:
        filetrain = "MNIST_mobile/{}/train/train.json".format(worker.client_id)
        os.makedirs(os.path.dirname(filetrain), mode=0o770, exist_ok=True)
        filetest = "MNIST_mobile/{}/test/test.json".format(worker.client_id)
        os.makedirs(os.path.dirname(filetest), mode=0o770, exist_ok=True)
        new_train["num_samples"] = [
            train_num_samples[i] for i in tuple(worker.client_sample_list)
        ]
        new_train["users"] = [users[i] for i in tuple(worker.client_sample_list)]
        client_sample = new_train["users"]
        new_train["user_data"] = {x: train_data[x] for x in client_sample}
        with open(filetrain, "w") as fp:
            json.dump(new_train, fp)
        new_test["num_samples"] = [
            test_num_samples[i] for i in tuple(worker.client_sample_list)
        ]
        new_test["users"] = [users[i] for i in tuple(worker.client_sample_list)]
        client_sample = new_test["users"]
        new_test["user_data"] = {x: test_data[x] for x in client_sample}
        with open(filetest, "w") as ff:
            json.dump(new_test, ff)
        shutil.make_archive(
            "MNIST_mobile/{}".format(worker.client_id),
            "zip",
            "MNIST_mobile",
            str(worker.client_id),
        )
        shutil.move("MNIST_mobile/{}.zip".format(worker.client_id), "MNIST_mobile_zip")
