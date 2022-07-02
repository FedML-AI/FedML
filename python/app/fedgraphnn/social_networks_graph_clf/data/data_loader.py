import copy
import random

import torch.utils.data as data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from fedml.core import partition_class_samples_with_dirichlet_distribution
from .datasets import MoleculesDataset
from .utils import *


def get_data(path, data, convert_x=True):

    tudataset = TUDataset(f"{path}", data)

    if not tudataset[0].__contains__("x") or convert_x:
        new_graphs = convert_to_nodeDegreeFeatures(tudataset)
        graphs = new_graphs
    else:
        graphs = [x for x in tudataset]

    return graphs, graphs[0].x.shape[1], tudataset.num_classes


def create_random_split(path, data):
    graphs, _, _ = get_data(path, data)

    graphs_tv, graphs_test = split_data(graphs, test=0.1, shuffle=True)

    graphs_train, graphs_val = split_data(graphs_tv, train=0.9, test=0.1, shuffle=True)

    return graphs_train, graphs_val, graphs_test


def create_non_uniform_split(args, idxs, client_number, is_train=True):
    logging.info("create_non_uniform_split------------------------------------------")
    N = len(idxs)
    alpha = args.partition_alpha
    logging.info("sample number = %d, client_number = %d" % (N, client_number))
    logging.info(idxs)
    idx_batch_per_client = [[] for _ in range(client_number)]
    (
        idx_batch_per_client,
        min_size,
    ) = partition_class_samples_with_dirichlet_distribution(
        N, alpha, client_number, idx_batch_per_client, idxs
    )
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
    graphs_train, graphs_val, graphs_test = create_random_split(path, args.dataset)

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


    global_data_dict = {"train": graphs_train, "val": graphs_val, "test": graphs_test}

    return global_data_dict, partition_dicts



# For centralized training
def get_dataloader(path, compact=True, normalize_features=False, normalize_adj=False):
    (
        train_adj_matrices,
        train_feature_matrices,
        train_labels,
        val_adj_matrices,
        val_feature_matrices,
        val_labels,
        test_adj_matrices,
        test_feature_matrices,
        test_labels,
    ) = create_random_split(path)

    train_dataset = MoleculesDataset(
        train_adj_matrices,
        train_feature_matrices,
        train_labels,
        path,
        compact=compact,
        split="train",
    )
    vaL_dataset = MoleculesDataset(
        val_adj_matrices,
        val_feature_matrices,
        val_labels,
        path,
        compact=compact,
        split="val",
    )
    test_dataset = MoleculesDataset(
        test_adj_matrices,
        test_feature_matrices,
        test_labels,
        path,
        compact=compact,
        split="test",
    )

    collator = (
        WalkForestCollator(normalize_features=normalize_features)
        if compact
        else DefaultCollator(
            normalize_features=normalize_features, normalize_adj=normalize_adj
        )
    )

    # IT IS VERY IMPORTANT THAT THE BATCH SIZE = 1. EACH BATCH IS AN ENTIRE MOLECULE.
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collator, pin_memory=True
    )
    val_dataloader = data.DataLoader(
        vaL_dataset, batch_size=1, shuffle=False, collate_fn=collator, pin_memory=True
    )
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collator, pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader


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


def load_partition_data_distributed(process_id, path, client_number, uniform=True):
    global_data_dict, partition_dicts = partition_data_by_sample_size(
        path, client_number, uniform
    )
    train_data_num = len(global_data_dict["train"])

    collator = WalkForestCollator(normalize_features=True)

    if process_id == 0:
        train_data_global = DataLoader(
            global_data_dict["train"],
            batch_size=32,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )
        val_data_global = DataLoader(
            global_data_dict["val"],
            batch_size=32,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )
        test_data_global = DataLoader(
            global_data_dict["test"],
            batch_size=32,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )

        train_data_local = None
        val_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        train_dataset_local = partition_dicts[process_id - 1]["train"]
        local_data_num = len(train_dataset_local)
        train_data_local = DataLoader(
            train_dataset_local,
            batch_size=32,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )
        val_data_local = DataLoader(
            partition_dicts[process_id - 1]["val"],
            batch_size=32,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )
        test_data_local = DataLoader(
            partition_dicts[process_id - 1]["test"],
            batch_size=32,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )
        train_data_global = None
        val_data_global = None
        test_data_global = None

    return (
        train_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        local_data_num,
        train_data_local,
        val_data_local,
        test_data_local,
    )
