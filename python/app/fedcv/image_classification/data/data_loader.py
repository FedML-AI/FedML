import os

import numpy as np
import torch
from .ImageNet.data_loader import load_partition_data_ImageNet
from .Landmarks.data_loader import load_partition_data_landmarks
from .cifar10.data_loader import load_partition_data_cifar10
from .cifar100.data_loader import load_partition_data_cifar100
from .FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from .MNIST.data_loader import load_partition_data_mnist, download_mnist
from .cinic10.data_loader import load_partition_data_cinic10
from .fed_cifar100.data_loader import load_partition_data_federated_cifar100
import logging


def load_data(args):
    return load_synthetic_data(args)


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def load_synthetic_data(args):
    dataset_name = str(args.dataset).lower()
    # check if the centralized training is enabled
    centralized = True if (args.client_num_in_total == 1 and args.training_type != "cross_silo") else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name == "mnist":
        download_mnist(args.data_cache_dir)
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_mnist(
            args,
            args.batch_size,
            train_path=args.data_cache_dir + "/MNIST/train",
            test_path=args.data_cache_dir + "/MNIST/test",
        )
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_emnist(args.dataset, args.data_cache_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_cifar100(args.dataset, args.data_cache_dir)
        args.client_num_in_total = client_num
    elif dataset_name in ["imagenet", "ILSVRC2012".lower()]:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_ImageNet(
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    elif dataset_name in ["gld23k", "landmarks"]:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_cache_dir, "mini_gld_train_split.csv")
        fed_test_map_file = os.path.join(args.data_cache_dir, "mini_gld_test.csv")

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_landmarks(
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            fed_train_map_file=fed_train_map_file,
            fed_test_map_file=fed_test_map_file,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_cache_dir, "federated_train.csv")
        fed_test_map_file = os.path.join(args.data_cache_dir, "test.csv")

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_landmarks(
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            fed_train_map_file=fed_train_map_file,
            fed_test_map_file=fed_test_map_file,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            raise ValueError("dataset_name %s not supported" % dataset_name)
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = data_loader(
            args.dataset,
            args.data_cache_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
        )

    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())
        }
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]
        }
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]
        }
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {
            cid: combine_batches(train_data_local_dict[cid]) for cid in train_data_local_dict.keys()
        }
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num
