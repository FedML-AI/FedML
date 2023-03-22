import logging
import pickle
import random

import torch.utils.data as data

from fedml.core import partition_class_samples_with_dirichlet_distribution
from .datasets import MoleculesDataset
from .utils import *


def get_data(path):
    with open(path + "/adjacency_matrices.pkl", "rb") as f:
        adj_matrices = pickle.load(f)

    with open(path + "/feature_matrices.pkl", "rb") as f:
        feature_matrices = pickle.load(f)

    labels = np.load(path + "/labels.npy")

    return adj_matrices, feature_matrices, labels

def create_random_split(path):
    adj_matrices, feature_matrices, labels = get_data(path)

    # Random 80/10/10 split as suggested in the MoleculeNet whitepaper
    train_range = (0, int(0.8 * len(adj_matrices)))
    val_range = (
        int(0.8 * len(adj_matrices)),
        int(0.8 * len(adj_matrices)) + int(0.1 * len(adj_matrices)),
    )
    test_range = (
        int(0.8 * len(adj_matrices)) + int(0.1 * len(adj_matrices)),
        len(adj_matrices),
    )

    all_idxs = list(range(len(adj_matrices)))
    random.shuffle(all_idxs)

    train_adj_matrices = [
        adj_matrices[all_idxs[i]] for i in range(train_range[0], train_range[1])
    ]
    train_feature_matrices = [
        feature_matrices[all_idxs[i]] for i in range(train_range[0], train_range[1])
    ]
    train_labels = [labels[all_idxs[i]] for i in range(train_range[0], train_range[1])]

    val_adj_matrices = [
        adj_matrices[all_idxs[i]] for i in range(val_range[0], val_range[1])
    ]
    val_feature_matrices = [
        feature_matrices[all_idxs[i]] for i in range(val_range[0], val_range[1])
    ]
    val_labels = [labels[all_idxs[i]] for i in range(val_range[0], val_range[1])]

    test_adj_matrices = [
        adj_matrices[all_idxs[i]] for i in range(test_range[0], test_range[1])
    ]
    test_feature_matrices = [
        feature_matrices[all_idxs[i]] for i in range(test_range[0], test_range[1])
    ]
    test_labels = [labels[all_idxs[i]] for i in range(test_range[0], test_range[1])]

    return (
        train_adj_matrices,
        train_feature_matrices,
        train_labels,
        val_adj_matrices,
        val_feature_matrices,
        val_labels,
        test_adj_matrices,
        test_feature_matrices,
        test_labels,
    )

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

    num_train_samples = len(train_adj_matrices)
    num_val_samples = len(val_adj_matrices)
    num_test_samples = len(test_adj_matrices)

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

        train_adj_matrices_client = [
            train_adj_matrices[idx] for idx in client_train_idxs
        ]
        train_feature_matrices_client = [
            train_feature_matrices[idx] for idx in client_train_idxs
        ]
        train_labels_client = [train_labels[idx] for idx in client_train_idxs]
        labels_of_all_clients.append(train_labels_client)

        val_adj_matrices_client = [val_adj_matrices[idx] for idx in client_val_idxs]
        val_feature_matrices_client = [
            val_feature_matrices[idx] for idx in client_val_idxs
        ]
        val_labels_client = [val_labels[idx] for idx in client_val_idxs]

        test_adj_matrices_client = [test_adj_matrices[idx] for idx in client_test_idxs]
        test_feature_matrices_client = [
            test_feature_matrices[idx] for idx in client_test_idxs
        ]
        test_labels_client = [test_labels[idx] for idx in client_test_idxs]

        train_dataset_client = MoleculesDataset(
            train_adj_matrices_client,
            train_feature_matrices_client,
            train_labels_client,
            path,
            compact=compact,
            split="train",
        )
        val_dataset_client = MoleculesDataset(
            val_adj_matrices_client,
            val_feature_matrices_client,
            val_labels_client,
            path,
            compact=compact,
            split="val",
        )
        test_dataset_client = MoleculesDataset(
            test_adj_matrices_client,
            test_feature_matrices_client,
            test_labels_client,
            path,
            compact=compact,
            split="test",
        )

        partition_dict = {
            "train": train_dataset_client,
            "val": val_dataset_client,
            "test": test_dataset_client,
        }

        partition_dicts[client] = partition_dict
    global_data_dict = {
        "train": MoleculesDataset(
            train_adj_matrices,
            train_feature_matrices,
            train_labels,
            path,
            compact=compact,
            split="train",
        ),
        "val": MoleculesDataset(
            val_adj_matrices,
            val_feature_matrices,
            val_labels,
            path,
            compact=compact,
            split="val",
        ),
        "test": MoleculesDataset(
            test_adj_matrices,
            test_feature_matrices,
            test_labels,
            path,
            compact=compact,
            split="test",
        ),
    }

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

    # IT IS VERY IMPORTANT THAT THE BATCH SIZE = 1. EACH BATCH IS AN ENTIRE MOLECULE.
    train_data_global = data.DataLoader(
        global_data_dict["train"],
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
    )
    val_data_global = data.DataLoader(
        global_data_dict["val"],
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
    )
    test_data_global = data.DataLoader(
        global_data_dict["test"],
        batch_size=1,
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
        train_data_local_dict[client] = data.DataLoader(
            train_dataset_client,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )
        val_data_local_dict[client] = data.DataLoader(
            val_dataset_client,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True,
        )
        test_data_local_dict[client] = (
            test_data_global
            if global_test
            else data.DataLoader(
                test_dataset_client,
                batch_size=1,
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
        train_data_global = data.DataLoader(
            global_data_dict["train"],
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )
        val_data_global = data.DataLoader(
            global_data_dict["val"],
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )
        test_data_global = data.DataLoader(
            global_data_dict["test"],
            batch_size=1,
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
        train_data_local = data.DataLoader(
            train_dataset_local,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )
        val_data_local = data.DataLoader(
            partition_dicts[process_id - 1]["val"],
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )
        test_data_local = data.DataLoader(
            partition_dicts[process_id - 1]["test"],
            batch_size=1,
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


def load_moleculenet(args, dataset_name):
    num_cats, feat_dim = 0, 0
    if dataset_name not in ["sider", "tox21", "muv","qm8" ]:
        raise Exception("no such dataset!")

    compact = args.model == "graphsage"

    _, feature_matrices, labels = get_data(args.data_dir + args.dataset)
    unif = True if args.partition_method == "homo" else False

    if args.dataset == "pcba":
        args.metric = "prc-auc"
    else:
        args.metric = "roc-auc"

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
        args.data_dir + args.dataset,
        args.client_num_in_total,
        uniform=unif,
        compact=compact,
        normalize_features=args.normalize_features,
        normalize_adj=args.normalize_adjacency,
    )

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        labels[0].shape[0],
    ]

    return dataset, feature_matrices[0].shape[1], labels[0].shape[0]