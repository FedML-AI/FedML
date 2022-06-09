import logging

from typing import Tuple, Callable, Optional, List, Iterable, Union, Literal, Sized

import numpy as np
import torch.utils.data as data
from torchvision import transforms

from fedml.core.data.noniid_partition import record_data_stats, non_iid_partition_with_dirichlet_distribution
from .dataset import CocoSegmentation
from .transforms import Normalize, ToTensor, FixedResize


def _data_transforms_coco_segmentation() -> Tuple[Callable, Callable]:
    COCO_MEAN = (0.485, 0.456, 0.406)
    COCO_STD = (0.229, 0.224, 0.225)

    transform = transforms.Compose([FixedResize(513), Normalize(mean=COCO_MEAN, std=COCO_STD), ToTensor()])

    return transform, transform


# for centralized training
def get_dataloader(
    _, data_dir: str, train_bs: int, test_bs: int, data_idxs: Optional[List[int]] = None
) -> Iterable[Union[data.DataLoader, int]]:
    return get_dataloader_coco_segmentation(data_dir, train_bs, test_bs, data_idxs)


# for local devices
def get_dataloader_test(
    data_dir: str,
    train_bs: int,
    test_bs: int,
    data_idxs_train: Optional[List[int]],
    data_idxs_test: Optional[List[int]],
) -> Iterable[Union[data.DataLoader, int]]:
    return get_dataloader_coco_segmentation_test(data_dir, train_bs, test_bs, data_idxs_train, data_idxs_test)


def get_dataloader_coco_segmentation(
    data_dir: str, train_bs: int, test_bs: int, data_idxs: Optional[List[int]] = None
) -> Iterable[Union[data.DataLoader, int]]:
    transform_train, transform_test = _data_transforms_coco_segmentation()

    train_ds = CocoSegmentation(
        data_dir, split="train", transform=transform_train, download_dataset=False, data_idxs=data_idxs
    )

    test_ds = CocoSegmentation(data_dir, split="val", transform=transform_test, download_dataset=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl, train_ds.num_classes


def get_dataloader_coco_segmentation_test(
    data_dir: str,
    train_bs: int,
    test_bs: int,
    data_idxs_train: Optional[List[int]] = None,
    data_idxs_test: Optional[List[int]] = None,
) -> Iterable[Union[data.DataLoader, int]]:
    transform_train, transform_test = _data_transforms_coco_segmentation()

    train_ds = CocoSegmentation(
        data_dir, split="train", transform=transform_train, download_dataset=False, data_idxs=data_idxs_train
    )

    test_ds = CocoSegmentation(
        data_dir, split="val", transform=transform_test, download_dataset=False, data_idxs=data_idxs_test
    )

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl, train_ds.num_classes


def load_coco_segmentation_data(data_dir: str) -> Iterable[Union[List[int], np.ndarray, Sized[str]]]:
    transform_train, transform_test = _data_transforms_coco_segmentation()

    train_ds = CocoSegmentation(data_dir, split="train", transform=transform_train, download_dataset=False)
    test_ds = CocoSegmentation(data_dir, split="val", transform=transform_test, download_dataset=False)

    return train_ds.img_ids, train_ds.target, train_ds.cat_ids, test_ds.img_ids, test_ds.target, test_ds.cat_ids


# Get a partition map for each client
def partition_data(data_dir: str, partition: Literal["homo", "hetero"], n_nets: int, alpha: float):
    logging.info("********************* Partitioning data **********************")

    net_data_idx_map = None
    train_images, train_targets, train_cat_ids, _, __, ___ = load_coco_segmentation_data(data_dir)
    n_train = len(train_images)  # Number of training samples

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)  # As many splits as n_nets = number of clients
        net_data_idx_map = {i: batch_idxs[i] for i in range(n_nets)}

    # non-iid data distribution
    # TODO: Add custom non-iid distribution option - hetero-fix
    elif partition == "hetero":
        categories = train_cat_ids  # category names
        net_data_idx_map = non_iid_partition_with_dirichlet_distribution(
            train_targets, n_nets, categories, alpha, task="segmentation"
        )

    train_data_cls_counts = record_data_stats(train_targets, net_data_idx_map, task="segmentation")

    return net_data_idx_map, train_data_cls_counts


def load_partition_data_distributed_coco_segmentation(
    process_id: int,
    dataset: CocoSegmentation,
    data_dir: str,
    partition_method: Literal["homo", "hetero"],
    partition_alpha: float,
    client_number: int,
    batch_size: int,
):
    net_data_idx_map, train_data_cls_counts = partition_data(data_dir, partition_method, client_number, partition_alpha)

    train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info(
            "Number of global train batches: {} and test batches: {}".format(
                len(train_data_global), len(test_data_global)
            )
        )

        train_data_local_dict = None
        test_data_local_dict = None
        data_local_num_dict = None
    else:
        # get local dataset
        client_id = process_id - 1
        data_idxs = net_data_idx_map[client_id]
        local_data_num = len(data_idxs)
        logging.info("Total number of local images: {} in client ID {}".format(local_data_num, process_id))

        train_data_local, test_data_local, class_num = get_dataloader(
            dataset, data_dir, batch_size, batch_size, data_idxs
        )
        logging.info(
            "Number of local train batches: {} and test batches: {} in client ID {}".format(
                len(train_data_local), len(test_data_local), process_id
            )
        )

        data_local_num_dict = {client_id: local_data_num}
        train_data_local_dict = {client_id: train_data_local}
        test_data_local_dict = {client_id: test_data_local}
        train_data_global = None
        test_data_global = None
    return (
        train_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )


# Called from main_fedseg
def load_partition_data_coco_segmentation(
    dataset: CocoSegmentation,
    data_dir: str,
    partition_method: Literal["homo", "hetero"],
    partition_alpha: float,
    client_number: int,
    batch_size: int,
):
    net_data_idx_map, train_data_cls_counts = partition_data(data_dir, partition_method, client_number, partition_alpha)

    train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])

    # Global train and test data
    train_data_global, test_data_global, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info(
        "Number of global train batches: {} and test batches: {}".format(len(train_data_global), len(test_data_global))
    )

    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        data_idxs = net_data_idx_map[client_idx]
        local_data_num = len(data_idxs)
        logging.info("Total number of local images: {} in client ID {}".format(local_data_num, client_idx))

        data_local_num_dict[client_idx] = local_data_num

        train_data_local, test_data_local, class_num = get_dataloader(
            dataset, data_dir, batch_size, batch_size, data_idxs
        )

        logging.info(
            "Number of local train images: {} and test images: {} in client ID {}".format(
                len(train_data_local), len(test_data_local), client_idx
            )
        )

        # Store data loaders for each client as they contain specific data
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
