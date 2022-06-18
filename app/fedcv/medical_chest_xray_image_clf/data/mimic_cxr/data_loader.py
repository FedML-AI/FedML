import logging

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

from .dataset import MIMICCXR


def _get_mean_and_std(dataset: Dataset):
    """Compute the mean and std of dataset."""
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for i, (img, _) in enumerate(data_loader):
        if i % 1000 == 0:
            print(i)
        mean += img.mean(dim=(0, 2, 3))
        std += img.std(dim=(0, 2, 3))
    mean /= len(data_loader)
    std /= len(data_loader)
    return mean, std


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
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


def _data_transforms_mimiccxr():

    MIMICCXR_MEAN = [0.503, 0.503, 0.503]
    MIMICCXR_STD = [0.291, 0.291, 0.291]

    image_size = 256
    train_transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MIMICCXR_MEAN, MIMICCXR_STD),
        ]
    )

    # train_transform.transforms.append(Cutout(16))

    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MIMICCXR_MEAN, MIMICCXR_STD),
        ]
    )

    return train_transform, test_transform


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_mimiccxr(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_mimiccxr(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_mimiccxr(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = MIMICCXR

    transform_train, transform_test = _data_transforms_mimiccxr()

    train_ds = dl_obj(
        datadir,
        dataidxs=dataidxs,
        train=True,
        transform=transform_train,
        download=False,
    )
    test_ds = dl_obj(
        datadir,
        dataidxs=None,
        train=False,
        transform=transform_test,
        download=False,
    )

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    return train_dl, test_dl


def get_dataloader_test_mimiccxr(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = MIMICCXR

    transform_train, transform_test = _data_transforms_mimiccxr()

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

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    return train_dl, test_dl


def distributed_centralized_mimiccxr_loader(dataset, data_dir, world_size, rank, batch_size):
    """
    Used for generating distributed dataloader for
    accelerating centralized training
    """

    train_bs = batch_size
    test_bs = batch_size

    transform_train, transform_test = _data_transforms_mimiccxr()
    train_dataset = MIMICCXR(data_dir=data_dir, dataidxs=None, train=True, transform=transform_train)
    test_dataset = MIMICCXR(data_dir=data_dir, dataidxs=None, train=False, transform=transform_test)

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


def load_partition_data_mimiccxr(
    data_dir,
    partition_method="random",
    partition_alpha=None,
    client_number=100,
    batch_size=10,
):
    transform_train, transform_test = _data_transforms_mimiccxr()

    train_dataset = MIMICCXR(data_dir=data_dir, dataidxs=None, train=True, transform=transform_train)
    test_dataset = MIMICCXR(data_dir=data_dir, dataidxs=None, train=False, transform=transform_test)

    # get local dataset
    if partition_method == "random":
        num_train_items = int(len(train_dataset) / client_number)
        num_test_items = int(len(test_dataset) / client_number)
        dict_client = {}
        all_train_idxs = list(range(len(train_dataset)))
        all_test_idxs = list(range(len(test_dataset)))
        for client_idx in range(client_number):
            dict_client[client_idx] = {}
            dict_client[client_idx]["train"] = set(np.random.choice(all_train_idxs, num_train_items, replace=False))
            dict_client[client_idx]["test"] = set(np.random.choice(all_test_idxs, num_test_items, replace=False))
            all_train_idxs = list(set(all_train_idxs) - dict_client[client_idx]["train"])
            all_test_idxs = list(set(all_test_idxs) - dict_client[client_idx]["test"])
        if len(all_train_idxs) > 0:
            all_client_idxs = list(range(client_number))
            np.random.shuffle(all_client_idxs)
            choiced_client_idxs = all_client_idxs[: len(all_train_idxs)]
            for idx, client_idx in enumerate(choiced_client_idxs):
                dict_client[client_idx]["train"].add(all_train_idxs[idx])
        if len(all_test_idxs) > 0:
            all_client_idxs = list(range(client_number))
            np.random.shuffle(all_client_idxs)
            choiced_client_idxs = all_client_idxs[: len(all_test_idxs)]
            for idx, client_idx in enumerate(choiced_client_idxs):
                dict_client[client_idx]["test"].add(all_test_idxs[idx])
    else:
        raise NotImplementedError

    # build dataloader
    train_dl = []
    test_dl = []
    for client_idx in range(client_number):
        train_data_idxs = list(dict_client[client_idx]["train"])
        test_data_idxs = list(dict_client[client_idx]["test"])
        train_dl_, test_dl_ = get_dataloader_test_mimiccxr(
            datadir=data_dir,
            dataidxs_train=train_data_idxs,
            dataidxs_test=test_data_idxs,
            train_bs=batch_size,
            test_bs=batch_size,
        )
        train_dl.append(train_dl_)
        test_dl.append(test_dl_)

        logging.info(f"Client {client_idx} train data num: {len(train_dl_)} test data num: {len(test_dl_)}")

    logging.info("Partition data done")
    # logging.info("Partition data for each client: {}".format(dict_client))

    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)
    train_data_global = train_dataset
    test_data_global = test_dataset
    data_local_num_dict = {
        client_idx: len(dict_client[client_idx]["train"]) + len(dict_client[client_idx]["test"])
        for client_idx in range(client_number)
    }
    train_data_local_dict = {client_idx: train_dl_ for client_idx, train_dl_ in enumerate(train_dl)}
    test_data_local_dict = {client_idx: test_dl_ for client_idx, test_dl_ in enumerate(test_dl)}
    class_num = train_dataset.num_classes

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
    data_path = os.path.join("D:\\", "dataset", "CheXpert", "CheXpert-v1.0-small")
    data = MIMICCXR(data_dir=data_path, transform=transforms.ToTensor())
    print(len(data))
    print(data[0][0])
    print(data[0][1])

    # mean, std = _get_mean_and_std(data)
    # print(mean, std)

    # train_transform, valid_transform = _data_transforms_chexpert()
    # print(train_transform)
    # print(valid_transform)

    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_mimiccxr(data_dir=data_path, client_number=10, batch_size=10)

    print(train_data_num, test_data_num, class_num)
