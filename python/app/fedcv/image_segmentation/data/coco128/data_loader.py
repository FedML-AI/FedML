import logging

from typing import Tuple, Callable, Optional, List, Iterable, Union, Literal, Sized

import numpy as np
import torch.utils.data as data
from torchvision import transforms

from fedml.core.data.noniid_partition import (
    record_data_stats,
    non_iid_partition_with_dirichlet_distribution,
)
from .datasets import CocoSegmentDataset
from .transforms import Normalize, ToTensor, FixedResize


def _data_transforms_coco128_segmentation() -> Tuple[Callable, Callable]:
    COCO_MEAN = (0.485, 0.456, 0.406)
    COCO_STD = (0.229, 0.224, 0.225)

    transform = transforms.Compose(
        [FixedResize(512), Normalize(mean=COCO_MEAN, std=COCO_STD), ToTensor()]
    )

    return transform, transform


# for centralized training
def get_dataloader(
    _,
    data_dir: str,
    train_bs: int,
    test_bs: int,
    data_idxs: Optional[List[int]] = None,
    test: bool = False,
) -> Iterable[Union[data.DataLoader, int]]:
    return get_dataloader_coco128_segmentation(data_dir, train_bs, test_bs, data_idxs)


# for local devices
def get_dataloader_test(
    data_dir: str,
    train_bs: int,
    test_bs: int,
    data_idxs_train: Optional[List[int]],
    data_idxs_test: Optional[List[int]],
) -> Iterable[Union[data.DataLoader, int]]:
    return get_dataloader_coco128_segmentation_test(
        data_dir, train_bs, test_bs, data_idxs_train, data_idxs_test
    )


def get_dataloader_coco128_segmentation(
    data_dir: str,
    train_bs: int,
    test_bs: int,
    data_idxs: Optional[List[int]] = None,
    test: bool = True,
) -> Iterable[Union[data.DataLoader, int]]:
    transform_train, transform_test = _data_transforms_coco128_segmentation()

    train_ds = CocoSegmentDataset(
        data_dir,
        train=True,
        transform=transform_train,
        data_idxs=data_idxs,
    )
    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True
    )

    if test:
        test_ds = CocoSegmentDataset(
            data_dir, train=False, transform=transform_test, data_idxs=data_idxs
        )
        test_dl = data.DataLoader(
            dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True
        )
    else:
        test_dl = None

    return train_dl, test_dl, train_ds.num_classes


def get_dataloader_coco128_segmentation_test(
    data_dir: str,
    train_bs: int,
    test_bs: int,
    data_idxs_train: Optional[List[int]] = None,
    data_idxs_test: Optional[List[int]] = None,
) -> Iterable[Union[data.DataLoader, int]]:
    transform_train, transform_test = _data_transforms_coco128_segmentation()

    train_ds = CocoSegmentDataset(
        data_dir,
        train=True,
        transform=transform_train,
        data_idxs=data_idxs_train,
    )

    test_ds = CocoSegmentDataset(
        data_dir,
        train=True,
        transform=transform_test,
        data_idxs=data_idxs_test,
    )

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True
    )
    test_dl = data.DataLoader(
        dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True
    )

    return train_dl, test_dl, train_ds.num_classes


# Get a partition map for each client
def partition_data(
    data_dir: str, partition: Literal["homo", "hetero"], n_nets: int, alpha: float
):
    logging.info("********************* Partitioning data **********************")
    n_train = 128  # Number of training samples

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(
            idxs, n_nets
        )  # As many splits as n_nets = number of clients
        net_data_idx_map = {i: batch_idxs[i] for i in range(n_nets)}

    # non-iid data distribution
    # TODO: Add custom non-iid distribution option - hetero-fix
    elif partition == "hetero":
        raise NotImplementedError("Hetero partition not implemented")

    return net_data_idx_map


def load_partition_data_distributed_coco128_segmentation(
    process_id: int,
    dataset: CocoSegmentDataset,
    data_dir: str,
    partition_method: Literal["homo", "hetero"],
    partition_alpha: float,
    client_number: int,
    batch_size: int,
):
    net_data_idx_map = partition_data(
        data_dir, partition_method, client_number, partition_alpha
    )

    train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global, class_num = get_dataloader(
            dataset, data_dir, batch_size, batch_size
        )
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
        logging.info(
            "Total number of local images: {} in client ID {}".format(
                local_data_num, process_id
            )
        )

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
def load_partition_data_coco128_segmentation(
    args,
    dataset,
    data_dir: str,
    partition_method: Literal["homo", "hetero"],
    partition_alpha: float,
    client_number: int,
    batch_size: int,
):

    net_data_idx_map = partition_data(
        data_dir, partition_method, client_number, partition_alpha
    )

    train_data_global, test_data_global = None, None
    train_data_num = 0
    test_data_num = 0

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    class_num = 80

    if args.process_id == 0:  # server
        pass
    else:
        client_idx = int(args.process_id) - 1
        dataidxs = net_data_idx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info(
            "client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num)
        )

        train_data_local, test_data_local, class_num = get_dataloader(
            dataset, data_dir, batch_size, batch_size, dataidxs
        )
        logging.info(
            "client_idx = %d, batch_num_train_local = %d"
            % (client_idx, len(train_data_local))
        )
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
