import logging
import numpy as np
import torch.utils.data as data
from torchvision import transforms

from fedml.core.data.noniid_partition import record_data_stats, non_iid_partition_with_dirichlet_distribution

from .dataset import PascalVocAugmentedSegmentation
from ..pascal_voc_augmented import transforms as custom_transforms

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _data_transforms_pascal_voc(image_size):
    PASCAL_VOC_MEAN = (0.485, 0.456, 0.406)
    PASCAL_VOC_STD = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose(
        [
            custom_transforms.RandomMirror(),
            custom_transforms.RandomScaleCrop(image_size, image_size),
            custom_transforms.RandomGaussianBlur(),
            custom_transforms.ToTensor(),
            custom_transforms.Normalize(mean=PASCAL_VOC_MEAN, std=PASCAL_VOC_STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            custom_transforms.FixedScaleCrop(image_size),
            custom_transforms.ToTensor(),
            custom_transforms.Normalize(mean=PASCAL_VOC_MEAN, std=PASCAL_VOC_STD),
        ]
    )

    return train_transform, val_transform


# for centralized training
def get_dataloader(_, data_dir, train_bs, test_bs, image_size, data_idxs=None):
    return get_dataloader_pascal_voc(data_dir, train_bs, test_bs, image_size, data_idxs)


# for local devices
def get_dataloader_test(data_dir, train_bs, test_bs, image_size, data_idxs_train=None, data_idxs_test=None):
    return get_dataloader_pascal_voc_test(data_dir, train_bs, test_bs, image_size, data_idxs_train, data_idxs_test)


def get_dataloader_pascal_voc(data_dir, train_bs, test_bs, image_size, data_idxs=None):
    transform_train, transform_test = _data_transforms_pascal_voc(image_size)

    train_ds = PascalVocAugmentedSegmentation(
        data_dir, split="train", download_dataset=False, transform=transform_train, data_idxs=data_idxs
    )

    test_ds = PascalVocAugmentedSegmentation(data_dir, split="val", download_dataset=False, transform=transform_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl, len(train_ds.classes)


def get_dataloader_pascal_voc_test(data_dir, train_bs, test_bs, image_size, data_idxs_train=None, data_idxs_test=None):
    transform_train, transform_test = _data_transforms_pascal_voc(image_size)

    train_ds = PascalVocAugmentedSegmentation(
        data_dir, split="train", download_dataset=False, transform=transform_train, data_idxs=data_idxs_train
    )

    test_ds = PascalVocAugmentedSegmentation(
        data_dir, split="val", download_dataset=False, transform=transform_test, data_idxs=data_idxs_test
    )

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl, len(train_ds.classes)


def load_pascal_voc_data(data_dir, image_size):
    transform_train, transform_test = _data_transforms_pascal_voc(image_size)

    train_ds = PascalVocAugmentedSegmentation(
        data_dir, split="train", download_dataset=False, transform=transform_train
    )
    test_ds = PascalVocAugmentedSegmentation(data_dir, split="val", download_dataset=False, transform=transform_test)

    return train_ds.images, train_ds.targets, train_ds.classes, test_ds.images, test_ds.targets, test_ds.classes


# Get a partition map for each client
def partition_data(data_dir, partition, n_nets, alpha, image_size):
    logging.info("********************* Partitioning data **********************")
    net_data_idx_map = None
    train_images, train_targets, train_categories, _, __, ___ = load_pascal_voc_data(data_dir, image_size)
    n_train = len(train_images)  # Number of training samples

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)  # As many splits as n_nets = number of clients
        net_data_idx_map = {i: batch_idxs[i] for i in range(n_nets)}

    # non-iid data distribution
    # TODO: Add custom non-iid distribution option - hetero-fix
    elif partition == "hetero":
        # This is useful if we allow custom category lists, currently done for consistency
        categories = [train_categories.index(c) for c in train_categories]
        net_data_idx_map = non_iid_partition_with_dirichlet_distribution(
            train_targets, n_nets, categories, alpha, task="segmentation"
        )

    train_data_cls_counts = record_data_stats(train_targets, net_data_idx_map, task="segmentation")

    return net_data_idx_map, train_data_cls_counts


def load_partition_data_distributed_pascal_voc(
    process_id, dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, image_size
):
    net_data_idx_map, train_data_cls_counts = partition_data(
        data_dir, partition_method, client_number, partition_alpha, image_size
    )

    train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global, class_num = get_dataloader(
            dataset, data_dir, batch_size, batch_size, image_size
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
        logging.info("Total number of local images: {} in client ID {}".format(local_data_num, process_id))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local, class_num = get_dataloader(
            dataset, data_dir, batch_size, batch_size, image_size, data_idxs
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
def load_partition_data_pascal_voc(
    dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, image_size
):
    net_data_idx_map, train_data_cls_counts = partition_data(
        data_dir, partition_method, client_number, partition_alpha, image_size
    )

    train_data_num = sum([len(net_data_idx_map[r]) for r in range(client_number)])

    # Global train and test data
    train_data_global, test_data_global, class_num = get_dataloader(
        dataset, data_dir, batch_size, batch_size, image_size
    )
    logging.info(
        "Number of global train batches: {} and test batches: {}".format(len(train_data_global), len(test_data_global))
    )

    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()  # Number of samples for each client
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        data_idxs = net_data_idx_map[client_idx]  # get dataId list for client generated using Dirichlet sampling
        local_data_num = len(data_idxs)  # How many samples does client have?
        logging.info("Total number of local images: {} in client ID {}".format(local_data_num, client_idx))

        data_local_num_dict[client_idx] = local_data_num

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local, class_num = get_dataloader(
            dataset, data_dir, batch_size, batch_size, image_size, data_idxs
        )
        logging.info(
            "Number of local train batches: {} and test batches: {} in client ID {}".format(
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
