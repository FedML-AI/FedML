import os

import numpy as np
import torch
from .cityscapes.data_loader import load_partition_data_cityscapes
from .coco.segmentation.data_loader import load_partition_data_coco_segmentation
from .pascal_voc_augmented.data_loader import load_partition_data_pascal_voc
from .coco128.data_loader import load_partition_data_coco128_segmentation
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
    if dataset_name == "cityscapes":
        # load cityscapes dataset
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_cityscapes(
            args=args,
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )
    elif dataset_name in ["coco_segmentation", "coco"]:
        # load coco dataset
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_coco_segmentation(
            args=args,
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )
    elif dataset_name == "coco128":
        # load coco128 dataset
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_coco128_segmentation(
            args=args,
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            partition_method=args.partition_method,
            partition_alpha=args.partition_alpha,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )
    elif dataset_name in ["pascal_voc", "pascal_voc_augmented"]:
        # load pascal voc dataset
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_pascal_voc(
            args=args,
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError("dataset %s is not supported" % dataset_name)

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
