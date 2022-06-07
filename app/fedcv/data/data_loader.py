import os

import numpy as np
import torch
from fedcv.data.chexpert.data_loader import load_partition_data_chexpert
from fedcv.data.imagenet.data_loader import load_partition_data_ImageNet
from fedcv.data.cityscapes.data_loader import load_partition_data_cityscapes
from fedcv.data.landmarks.data_loader import load_partition_data_landmarks
import logging


def load(args):
    return load_synthetic_data(args)


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def load_synthetic_data(args):
    dataset_name = args.dataset
    # check if the centralized training is enabled
    centralized = True if (args.client_num_in_total == 1 and args.training_type != "cross_silo") else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name.lower() in ["chexpert", "chexpert_small"]:
        # load chexpert dataset
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
        ) = load_partition_data_chexpert(
            data_dir=args.data_cache_dir,
            partition_method="random",
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )
    elif dataset_name.lower() == "imagenet":
        # load imagenet dataset
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            data_local_num_dict,
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
    elif dataset_name.lower() == "cityscapes":
        # load cityscapes dataset
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_cityscapes(
            dataset=args.dataset,
            data_dir=args.data_dir,
            partition_method=args.partition_method,
            partition_alpha=args.partition_alpha,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
    elif dataset_name.lower() == "landmark":
        # load landmark dataset
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_landmarks(
            dataset=args.dataset,
            data_dir=args.data_dir,
            fed_train_map_file=args.fed_train_map_file,
            fed_test_map_file=args.fed_test_map_file,
            partition_method=args.partition_method,
            partition_alpha=args.partition_alpha,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )
    else:
        return None

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
