import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler


from .datasets import ImageNet
from .datasets import ImageNet_truncated
from .datasets_hdf5 import ImageNet_hdf5
from .datasets_hdf5 import ImageNet_truncated_hdf5


import numpy as np
import torch

class Cutout(object):
    """
    Apply the Cutout data augmentation technique to an image.

    Cutout is a technique used for regularization during training deep neural networks.
    It randomly masks out a rectangular region of the input image.

    Args:
        length (int): The length of the square mask to apply.

    Usage:
        transform = Cutout(length=16)  # Create an instance of the Cutout transform.
        transformed_image = transform(input_image)  # Apply the Cutout transform to an image.

    Note:
        The Cutout transform is typically applied as part of a data augmentation pipeline.

    References:
        - Original paper: https://arxiv.org/abs/1708.04552

    """

    def __init__(self, length):
        """
        Initialize the Cutout transform with the specified length.

        Args:
            length (int): The length of the square mask to apply.
        """
        self.length = length

    def __call__(self, img):
        """
        Apply Cutout transformation to an input image.

        Args:
            img (torch.Tensor): The input image tensor to which Cutout will be applied.

        Returns:
            torch.Tensor: The input image with a randomly masked region.

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


def _data_transforms_ImageNet():
    """
    Define data transforms for the ImageNet dataset.

    Returns:
        transforms.Compose: A composition of data augmentation transforms for training
        and validation data.
    """
    # IMAGENET_MEAN = [0.5071, 0.4865, 0.4409]
    # IMAGENET_STD = [0.2673, 0.2564, 0.2762]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    image_size = 224
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, valid_transform

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    """
    Get data loaders for centralized training.

    Args:
        dataset (str): The dataset name.
        datadir (str): The path to the dataset directory.
        train_bs (int): Batch size for training data.
        test_bs (int): Batch size for testing data.
        dataidxs (list, optional): List of data indices to use for training (default: None).

    Returns:
        DataLoader: Training and testing data loaders.
    """
    return get_dataloader_ImageNet(datadir, train_bs, test_bs, dataidxs)

def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    """
    Get data loaders for local devices.

    Args:
        dataset (str): The dataset name.
        datadir (str): The path to the dataset directory.
        train_bs (int): Batch size for training data.
        test_bs (int): Batch size for testing data.
        dataidxs_train (list): List of data indices to use for training.
        dataidxs_test (list): List of data indices to use for testing.

    Returns:
        DataLoader: Training and testing data loaders.
    """
    return get_dataloader_test_ImageNet(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_ImageNet_truncated(
    imagenet_dataset_train,
    imagenet_dataset_test,
    train_bs,
    test_bs,
    dataidxs=None,
    net_dataidx_map=None,
):
    """
    Get data loaders for a truncated version of the ImageNet dataset.

    Args:
        imagenet_dataset_train: The training dataset (ImageNet or ImageNet_hdf5).
        imagenet_dataset_test: The testing dataset (ImageNet or ImageNet_hdf5).
        train_bs (int): Batch size for training data.
        test_bs (int): Batch size for testing data.
        dataidxs (list, optional): List of data indices to use for training (default: None).
        net_dataidx_map (dict, optional): Mapping of data indices to network indices (default: None).

    Returns:
        tuple: A tuple containing training and testing data loaders.

    Raises:
        NotImplementedError: If the dataset type is not supported.

    Note:
        - The `imagenet_dataset_train` and `imagenet_dataset_test` should be instances of `ImageNet` or `ImageNet_hdf5`.

    """
    if type(imagenet_dataset_train) == ImageNet:
        dl_obj = ImageNet_truncated
    elif type(imagenet_dataset_train) == ImageNet_hdf5:
        dl_obj = ImageNet_truncated_hdf5
    else:
        raise NotImplementedError()

    transform_train, transform_test = _data_transforms_ImageNet()

    train_ds = dl_obj(
        imagenet_dataset_train,
        dataidxs,
        net_dataidx_map,
        train=True,
        transform=transform_train,
        download=False,
    )
    test_ds = dl_obj(
        imagenet_dataset_test,
        dataidxs=None,
        net_dataidx_map=None,
        train=False,
        transform=transform_test,
        download=False,
    )

    train_dl = data.DataLoader(
        dataset=train_ds,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )
    test_dl = data.DataLoader(
        dataset=test_ds,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    return train_dl, test_dl


def get_dataloader_ImageNet(datadir, train_bs, test_bs, dataidxs=None):
    """
    Get data loaders for the ImageNet dataset.

    Args:
        datadir (str): The path to the dataset directory.
        train_bs (int): Batch size for training data.
        test_bs (int): Batch size for testing data.
        dataidxs (list, optional): List of data indices to use for training (default: None).

    Returns:
        tuple: A tuple containing training and testing data loaders.

    """
    dl_obj = ImageNet

    transform_train, transform_test = _data_transforms_ImageNet()

    train_ds = dl_obj(
        datadir,
        dataidxs=dataidxs,
        train=True,
        transform=transform_train,
        download=False,
    )
    test_ds = dl_obj(
        datadir, dataidxs=None, train=False, transform=transform_test, download=False
    )

    train_dl = data.DataLoader(
        dataset=train_ds,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )
    test_dl = data.DataLoader(
        dataset=test_ds,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    return train_dl, test_dl


def get_dataloader_test_ImageNet(
    datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None
):
    """
    Get data loaders for the ImageNet dataset for testing.

    Args:
        datadir (str): The path to the dataset directory.
        train_bs (int): Batch size for training data.
        test_bs (int): Batch size for testing data.
        dataidxs_train (list, optional): List of data indices to use for training (default: None).
        dataidxs_test (list, optional): List of data indices to use for testing (default: None).

    Returns:
        tuple: A tuple containing training and testing data loaders.

    """
    dl_obj = ImageNet

    transform_train, transform_test = _data_transforms_ImageNet()

    train_ds = dl_obj(
        datadir,
        dataidxs=dataidxs_train,
        train=True,
        transform=transform_train,
        download=True,
    )
    test_ds = dl_obj(
        datadir,
        dataidxs=dataidxs_test,
        train=False,
        transform=transform_test,
        download=True,
    )

    train_dl = data.DataLoader(
        dataset=train_ds,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )
    test_dl = data.DataLoader(
        dataset=test_ds,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    return train_dl, test_dl


def distributed_centralized_ImageNet_loader(
    dataset, data_dir, world_size, rank, batch_size
):
    """
    Generate a distributed dataloader for accelerating centralized training.

    Args:
        dataset (str): The dataset name ("ILSVRC2012" or "ILSVRC2012_hdf5").
        data_dir (str): The path to the dataset directory.
        world_size (int): The total number of processes in the distributed training.
        rank (int): The rank of the current process in the distributed training.
        batch_size (int): Batch size for training and testing data.

    Returns:
        tuple: A tuple containing various training and testing data related information.

    """

    train_bs = batch_size
    test_bs = batch_size

    transform_train, transform_test = _data_transforms_ImageNet()
    
    if dataset == "ILSVRC2012":
        train_dataset = ImageNet(
            data_dir=data_dir, dataidxs=None, train=True, transform=transform_train
        )

        test_dataset = ImageNet(
            data_dir=data_dir, dataidxs=None, train=False, transform=transform_test
        )
    elif dataset == "ILSVRC2012_hdf5":
        train_dataset = ImageNet_hdf5(
            data_dir=data_dir, dataidxs=None, train=True, transform=transform_train
        )

        test_dataset = ImageNet_hdf5(
            data_dir=data_dir, dataidxs=None, train=False, transform=transform_test
        )

    train_sam = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sam = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_dl = data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        sampler=train_sam,
        pin_memory=True,
        num_workers=4,
    )
    test_dl = data.DataLoader(
        test_dataset,
        batch_size=test_bs,
        sampler=test_sam,
        pin_memory=True,
        num_workers=4,
    )

    class_num = 1000

    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)

    return train_data_num, test_data_num, train_dl, test_dl, None, None, None, class_num


def load_partition_data_ImageNet(
    dataset,
    data_dir,
    partition_method=None,
    partition_alpha=None,
    client_number=100,
    batch_size=10,
):
    """
    Load and partition data for the ImageNet dataset.

    Args:
        dataset (str): The dataset name ("ILSVRC2012" or "ILSVRC2012_hdf5").
        data_dir (str): The path to the dataset directory.
        partition_method (str, optional): The partitioning method (default: None).
        partition_alpha (float, optional): The partitioning alpha value (default: None).
        client_number (int, optional): The number of clients (default: 100).
        batch_size (int, optional): Batch size for training and testing data (default: 10).

    Returns:
        tuple: A tuple containing various data-related information.

    """

    if dataset == "ILSVRC2012":
        train_dataset = ImageNet(data_dir=data_dir, dataidxs=None, train=True)

        test_dataset = ImageNet(data_dir=data_dir, dataidxs=None, train=False)
    elif dataset == "ILSVRC2012_hdf5":
        train_dataset = ImageNet_hdf5(data_dir=data_dir, dataidxs=None, train=True)

        test_dataset = ImageNet_hdf5(data_dir=data_dir, dataidxs=None, train=False)

    net_dataidx_map = train_dataset.get_net_dataidx_map()

    class_num = 1000

    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    # train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)
    class_num_dict = train_dataset.get_data_local_num_dict()

    # train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)

    train_data_global, test_data_global = get_dataloader_ImageNet_truncated(
        train_dataset,
        test_dataset,
        train_bs=batch_size,
        test_bs=batch_size,
        dataidxs=None,
        net_dataidx_map=None,
    )

    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        if client_number == 1000:
            dataidxs = client_idx
            data_local_num_dict = class_num_dict
        elif client_number == 100:
            dataidxs = [client_idx * 10 + i for i in range(10)]
            data_local_num_dict[client_idx] = sum(
                class_num_dict[client_idx + i] for i in range(10)
            )
        else:
            raise NotImplementedError("Not support other client_number for now!")

        local_data_num = data_local_num_dict[client_idx]

        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        # train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
        #                                          dataidxs)
        train_data_local, test_data_local = get_dataloader_ImageNet_truncated(
            train_dataset,
            test_dataset,
            train_bs=batch_size,
            test_bs=batch_size,
            dataidxs=dataidxs,
            net_dataidx_map=net_dataidx_map,
        )

        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        # client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    logging.info("data_local_num_dict: %s" % data_local_num_dict)
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


if __name__ == "__main__":
    # data_dir = '/home/datasets/imagenet/ILSVRC2012_dataset'
    data_dir = "/home/datasets/imagenet/imagenet_hdf5/imagenet-shuffled.hdf5"

    client_number = 100
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
        None,
        data_dir,
        partition_method=None,
        partition_alpha=None,
        client_number=client_number,
        batch_size=10,
    )

    print(train_data_num, test_data_num, class_num)
    print(data_local_num_dict)

    print(train_data_num, test_data_num, class_num)
    print(data_local_num_dict)

    i = 0
    for data, label in train_data_global:
        print(data)
        print(label)
        i += 1
        if i > 5:
            break
    print("=============================\n")

    for client_idx in range(client_number):
        i = 0
        for data, label in train_data_local_dict[client_idx]:
            print(data)
            print(label)
            i += 1
            if i > 5:
                break
