import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import ImageFolderTruncated


# generate the non-IID distribution for all methods
def read_data_distribution(
    filename="./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt",
):
    """
    Reads the data distribution from a text file.

    Args:
        filename (str): The path to the distribution file.

    Returns:
        dict: A dictionary representing the data distribution.
    """
    distribution = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0]:
                tmp = x.split(":")
                if "{" == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(
                        tmp[1].strip().replace(",", "")
                    )
    return distribution


def read_net_dataidx_map(
    filename="./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt",
):
    """
    Reads the network data index map from a text file.

    Args:
        filename (str): The path to the network data index map file.

    Returns:
        dict: A dictionary mapping network IDs to data indices.
    """
    net_dataidx_map = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0] and "]" != x[0]:
                tmp = x.split(":")
                if "[" == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(",")
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    """
    Records network-specific data statistics.

    Args:
        y_train (numpy.ndarray): Array of ground truth labels for the entire dataset.
        net_dataidx_map (dict): A dictionary mapping network IDs to data indices.

    Returns:
        dict: A dictionary containing network-specific class counts.
    """
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """
        Applies the Cutout augmentation to an image.

        Args:
            img (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The image with the Cutout augmentation applied.
        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cinic10():
    """
    Define data transformations for the CIFAR-10 dataset.

    Returns:
        tuple: A tuple containing two transformation functions, one for training and one for validation/test.
    """
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    # Transformer for train set: random crops and horizontal flip
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: F.pad(
                    x.unsqueeze(0), (4, 4, 4, 4), mode="reflect"
                ).data.squeeze()
            ),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std),
        ]
    )

    # Transformer for test set
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: F.pad(
                    x.unsqueeze(0), (4, 4, 4, 4), mode="reflect"
                ).data.squeeze()
            ),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std),
        ]
    )
    return train_transform, valid_transform


def load_cinic10_data(datadir):
    """
    Load CIFAR-10 data from the specified directory.

    Args:
        datadir (str): The directory containing CIFAR-10 data.

    Returns:
        tuple: A tuple containing training and testing data.
    """
    _train_dir = datadir + str("/train")
    logging.info("_train_dir = " + str(_train_dir))
    _test_dir = datadir + str("/test")
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    trainset = ImageFolderTruncated(
        _train_dir,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: F.pad(
                        x.unsqueeze(0), (4, 4, 4, 4), mode="reflect"
                    ).data.squeeze()
                ),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=cinic_mean, std=cinic_std),
            ]
        ),
    )

    testset = ImageFolderTruncated(
        _test_dir,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: F.pad(
                        x.unsqueeze(0), (4, 4, 4, 4), mode="reflect"
                    ).data.squeeze()
                ),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=cinic_mean, std=cinic_std),
            ]
        ),
    )
    X_train, y_train = trainset.imgs, trainset.targets
    X_test, y_test = testset.imgs, testset.targets
    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha):
    """
    Partition the dataset into subsets for federated learning.

    Args:
        dataset: The dataset to be partitioned.
        datadir (str): The directory containing the dataset.
        partition (str): The type of partitioning to be applied ("homo", "hetero", "hetero-fix").
        n_nets (int): The number of clients (networks) to partition the data for.
        alpha (float): A hyperparameter controlling the heterogeneity of the data partition.

    Returns:
        tuple: A tuple containing partitioned data and related information.
    """
    logging.info("*********partition data***************")

    X_train, y_train, X_test, y_test = load_cinic10_data(datadir)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    n_train = len(X_train)


    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]

            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))

                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_nets)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = (
            "./data_preprocessing/non-iid-distribution/CINIC10/net_dataidx_map.txt"
        )
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = (
            "./data_preprocessing/non-iid-distribution/CINIC10/distribution.txt"
        )
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts



def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    """
    Get data loaders for centralized training using the CIFAR-10 dataset.

    Args:
        dataset (str): The dataset name.
        datadir (str): The directory containing the dataset.
        train_bs (int): Batch size for training data.
        test_bs (int): Batch size for testing data.
        dataidxs (list, optional): List of data indices to use for training. Default is None.

    Returns:
        tuple: A tuple containing the training and testing data loaders.
    """
    return get_dataloader_cinic10(datadir, train_bs, test_bs, dataidxs)



def get_dataloader_test(
    dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test
):
    """
    Get data loaders for decentralized (local devices) testing using the CIFAR-10 dataset.

    Args:
        dataset (str): The dataset name.
        datadir (str): The directory containing the dataset.
        train_bs (int): Batch size for training data.
        test_bs (int): Batch size for testing data.
        dataidxs_train (list): List of data indices to use for training.
        dataidxs_test (list): List of data indices to use for testing.

    Returns:
        tuple: A tuple containing the training and testing data loaders for local devices.
    """
    return get_dataloader_test_cinic10(
        datadir, train_bs, test_bs, dataidxs_train, dataidxs_test
    )


def get_dataloader_cinic10(datadir, train_bs, test_bs, dataidxs=None):
    """
    Get data loaders for centralized training using the CIFAR-10 dataset.

    Args:
        datadir (str): The directory containing the dataset.
        train_bs (int): Batch size for training data.
        test_bs (int): Batch size for testing data.
        dataidxs (list, optional): List of data indices to use for training. Default is None.

    Returns:
        tuple: A tuple containing the training and testing data loaders.
    """
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_cinic10()

    traindir = os.path.join(datadir, "train")
    valdir = os.path.join(datadir, "test")

    train_ds = dl_obj(traindir, dataidxs=dataidxs, transform=transform_train)
    test_ds = dl_obj(valdir, transform=transform_train)

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True
    )
    test_dl = data.DataLoader(
        dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True
    )

    return train_dl, test_dl


def get_dataloader_test_cinic10(
    datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None
):
    """
    Get data loaders for decentralized (local devices) testing using the CIFAR-10 dataset.

    Args:
        datadir (str): The directory containing the dataset.
        train_bs (int): Batch size for training data.
        test_bs (int): Batch size for testing data.
        dataidxs_train (list): List of data indices to use for training.
        dataidxs_test (list): List of data indices to use for testing.

    Returns:
        tuple: A tuple containing the training and testing data loaders for local devices.
    """
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_cinic10()

    traindir = os.path.join(datadir, "train")
    valdir = os.path.join(datadir, "test")

    train_ds = dl_obj(traindir, dataidxs=dataidxs_train, transform=transform_train)
    test_ds = dl_obj(valdir, dataidxs=dataidxs_test, transform=transform_test)

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True
    )
    test_dl = data.DataLoader(
        dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True
    )

    return train_dl, test_dl


def load_partition_data_distributed_cinic10(
    process_id,
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
):
    """
    Load partitioned data for distributed training using the CIFAR-10 dataset.

    Args:
        process_id (int): The ID of the current process.
        dataset (str): The dataset name.
        data_dir (str): The directory containing the dataset.
        partition_method (str): The data partitioning method (e.g., 'homo' or 'hetero').
        partition_alpha (float): The alpha parameter for data partitioning.
        client_number (int): The number of clients.
        batch_size (int): Batch size for data loaders.

    Returns:
        tuple: A tuple containing training and testing data information for distributed training.
    """
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        traindata_cls_counts,
    ) = partition_data(
        dataset, data_dir, partition_method, client_number, partition_alpha
    )
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(
            dataset, data_dir, batch_size, batch_size
        )
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        test_data_num = len(test_data_global)
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info(
            "rank = %d, local_sample_number = %d" % (process_id, local_data_num)
        )
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, dataidxs
        )
        logging.info(
            "process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (process_id, len(train_data_local), len(test_data_local))
        )
        test_data_num = 0
        train_data_global = None
        test_data_global = None

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        local_data_num,
        train_data_local,
        test_data_local,
        class_num,
    )


def load_partition_data_cinic10(
    dataset, data_dir, partition_method, partition_alpha, client_number, batch_size
):
    """
    Load partitioned data for centralized training using the CIFAR-10 dataset.

    Args:
        dataset (str): The dataset name.
        data_dir (str): The directory containing the dataset.
        partition_method (str): The data partitioning method (e.g., 'homo' or 'hetero').
        partition_alpha (float): The alpha parameter for data partitioning.
        client_number (int): The number of clients.
        batch_size (int): Batch size for data loaders.

    Returns:
        tuple: A tuple containing training and testing data information for centralized training.
    """
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        traindata_cls_counts,
    ) = partition_data(
        dataset, data_dir, partition_method, client_number, partition_alpha
    )
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(
        dataset, data_dir, batch_size, batch_size
    )
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info(
            "client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num)
        )

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, dataidxs
        )
        logging.info(
            "client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (client_idx, len(train_data_local), len(test_data_local))
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
