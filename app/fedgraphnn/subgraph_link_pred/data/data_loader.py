import os
import random
import copy
import logging
import pickle
import numpy as np

from torch_geometric.data import DataLoader
from .utils import DefaultCollator, WalkForestCollator

from fedml.core import partition_class_samples_with_dirichlet_distribution


def get_data_community(path, data, pred_task, algo="Louvain"):
    assert pred_task in ["relation", "link"]
    assert data in ["wn18rr", "FB15k-237", "YAGO3-10"]

    logging.info("-------------------")
    num_of_classes = 2

    graphs_train = pickle.load(
        open(os.path.join(path, data, "train.pkl"), "rb")
    )
    graphs_val = pickle.load(open(os.path.join(path, data, "valid.pkl"), "rb"))
    graphs_test = pickle.load(open(os.path.join(path, data, "test.pkl"), "rb"))

    # number of graphs == number of relation type
    return graphs_train, graphs_val, graphs_test, num_of_classes


def create_random_split(path, data, pred_task="link", algo="Louvain"):
    assert pred_task in ["relation", "link"]

    graphs_train, graphs_val, graphs_test, num_of_classes = get_data_community(
        path, data, pred_task, algo
    )

    return graphs_train, graphs_val, graphs_test, num_of_classes


def create_non_uniform_split(args, idxs, client_number, is_train=True):
    logging.info("create_non_uniform_split------------------------------------------")
    N = len(idxs)

    min_size = 0
    alpha = args.partition_alpha
    logging.info("sample number = %d, client_number = %d" % (N, client_number))
    logging.info(idxs)
    while min_size < 1:
        idx_batch_per_client = [[] for _ in range(client_number)]
        (
            idx_batch_per_client,
            min_size,
        ) = partition_class_samples_with_dirichlet_distribution(
            N, alpha, client_number, idx_batch_per_client, idxs
        )
        logging.info("searching for min_size < 1")
    logging.info(idx_batch_per_client)
    sample_num_distribution = []

    for client_id in range(client_number):
        sample_num_distribution.append(len(idx_batch_per_client[client_id]))
        logging.info(
            "client_id = %d, sample_number = %d"
            % (client_id, len(idx_batch_per_client[client_id]))
        )
    logging.info("create_non_uniform_split******************************************")

    return idx_batch_per_client


def partition_data_by_sample_size(
    args, path, client_number, uniform=True, compact=True
):
    graphs_train, graphs_val, graphs_test, num_of_classes = create_random_split(
        path, args.dataset, args.pred_task, args.part_algo
    )

    num_train_samples = len(graphs_train)
    num_val_samples = len(graphs_val)
    num_test_samples = len(graphs_test)

    train_idxs = list(range(num_train_samples))
    val_idxs = list(range(num_val_samples))
    test_idxs = list(range(num_test_samples))

    random.shuffle(train_idxs)
    random.shuffle(val_idxs)
    random.shuffle(test_idxs)

    partition_dicts = [None] * client_number

    if uniform:
        clients_idxs_train = np.array_split(train_idxs, client_number)
        clients_idxs_val = np.array_split(val_idxs, client_number)
        clients_idxs_test = np.array_split(test_idxs, client_number)
    else:
        clients_idxs_train = create_non_uniform_split(
            args, train_idxs, client_number, True
        )
        clients_idxs_val = create_non_uniform_split(
            args, val_idxs, client_number, False
        )
        clients_idxs_test = create_non_uniform_split(
            args, test_idxs, client_number, False
        )

    labels_of_all_clients = []
    for client in range(client_number):
        client_train_idxs = clients_idxs_train[client]
        client_val_idxs = clients_idxs_val[client]
        client_test_idxs = clients_idxs_test[client]

        train_graphs_client = [graphs_train[idx] for idx in client_train_idxs]
        train_labels_client = [graphs_train[idx].y for idx in client_train_idxs]
        labels_of_all_clients.append(train_labels_client)

        val_graphs_client = [graphs_val[idx] for idx in client_val_idxs]

        val_labels_client = [graphs_val[idx].y for idx in client_val_idxs]
        labels_of_all_clients.append(val_labels_client)

        test_graphs_client = [graphs_test[idx] for idx in client_test_idxs]

        test_labels_client = [graphs_test[idx].y for idx in client_test_idxs]
        labels_of_all_clients.append(test_labels_client)

        partition_dict = {
            "train": train_graphs_client,
            "val": val_graphs_client,
            "test": test_graphs_client,
        }

        partition_dicts[client] = partition_dict

    # plot the label distribution similarity score
    # visualize_label_distribution_similarity_score(labels_of_all_clients)

    global_data_dict = {"train": graphs_train, "val": graphs_val, "test": graphs_test}

    return global_data_dict, partition_dicts


def visualize_label_distribution_similarity_score(labels_of_all_clients):
    label_distribution_clients = []
    label_num = labels_of_all_clients[0][0]
    for client_idx in range(len(labels_of_all_clients)):
        labels_client_i = labels_of_all_clients[client_idx]
        sample_number = len(labels_client_i)
        active_property_count = [0.0] * label_num
        for sample_index in range(sample_number):
            label = labels_client_i[sample_index]
            for property_index in range(len(label)):
                # logging.info(label[property_index])
                if label[property_index] == 1:
                    active_property_count[property_index] += 1
        active_property_count = [
            float(active_property_count[i]) for i in range(len(active_property_count))
        ]
        label_distribution_clients.append(copy.deepcopy(active_property_count))
    logging.info(label_distribution_clients)

    client_num = len(label_distribution_clients)
    label_distribution_similarity_score_matrix = np.random.random(
        (client_num, client_num)
    )

    for client_i in range(client_num):
        label_distribution_client_i = label_distribution_clients[client_i]
        for client_j in range(client_i, client_num):
            label_distribution_client_j = label_distribution_clients[client_j]
            logging.info(label_distribution_client_i)
            logging.info(label_distribution_client_j)
            a = np.array(label_distribution_client_i, dtype=np.float32)
            b = np.array(label_distribution_client_j, dtype=np.float32)

            from scipy import spatial

            distance = 1 - spatial.distance.cosine(a, b)
            label_distribution_similarity_score_matrix[client_i][client_j] = distance
            label_distribution_similarity_score_matrix[client_j][client_i] = distance
        # break
    logging.info(label_distribution_similarity_score_matrix)


# Single process sequential
def load_partition_data(
    args,
    path,
    client_number,
    uniform=True,
    global_test=True,
    compact=True,
    normalize_features=False,
    normalize_adj=False,
):
    global_data_dict, partition_dicts = partition_data_by_sample_size(
        args, path, client_number, uniform, compact=compact
    )

    data_local_num_dict = dict()
    train_data_local_dict = dict()
    val_data_local_dict = dict()
    test_data_local_dict = dict()

    collator = (
        WalkForestCollator(normalize_features=normalize_features)
        if compact
        else DefaultCollator(
            normalize_features=normalize_features, normalize_adj=normalize_adj
        )
    )

    # This is a PyG Dataloader
    train_data_global = DataLoader(
        global_data_dict["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
    )
    val_data_global = DataLoader(
        global_data_dict["val"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
    )
    test_data_global = DataLoader(
        global_data_dict["test"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
    )

    train_data_num = len(global_data_dict["train"])
    val_data_num = len(global_data_dict["val"])
    test_data_num = len(global_data_dict["test"])

    for client in range(client_number):
        train_dataset_client = partition_dicts[client]["train"]
        val_dataset_client = partition_dicts[client]["val"]
        test_dataset_client = partition_dicts[client]["test"]

        data_local_num_dict[client] = len(train_dataset_client)
        train_data_local_dict[client] = DataLoader(
            train_dataset_client,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )
        val_data_local_dict[client] = DataLoader(
            val_dataset_client,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True,
        )
        test_data_local_dict[client] = (
            test_data_global
            if global_test
            else DataLoader(
                test_dataset_client,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collator,
                pin_memory=True,
            )
        )

        logging.info(
            "Client idx = {}, local sample number = {}".format(
                client, len(train_dataset_client)
            )
        )

    return (
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
    )


def load_subgraph_data(args, dataset_name):
    if args.dataset not in ["YAGO3-10", "wn18rr", "FB15k-237"]:
        raise Exception("no such dataset!")

    args.part_algo = "Louvain"
    args.pred_task = "relation"
    compact = args.model == "graphsage"

    args.metric = "AP"

    g1, _, _ = get_data_community(args.data_dir, args.dataset, args.pred_task)
    print(g1[0])
    unif = True if args.partition_method == "homo" else False

    if args.model == "gcn":
        args.normalize_features = True
        args.normalize_adjacency = True

    (
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
    ) = load_partition_data(
        args,
        args.data_dir,
        args.client_num_in_total,
        uniform=unif,
        compact=compact,
        normalize_features=args.normalize_features,
        normalize_adj=args.normalize_adjacency,
    )

    dataset = [
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
    ]

    return dataset
