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


def _data_transforms_ImageNet():
    # IMAGENET_MEAN = [0.5071, 0.4865, 0.4409]
    # IMAGENET_STD = [0.2673, 0.2564, 0.2762]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    image_size = 224
    train_transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, valid_transform


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_ImageNet(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(
    dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test
):
    return get_dataloader_test_ImageNet(
        datadir, train_bs, test_bs, dataidxs_train, dataidxs_test
    )


def get_dataloader_ImageNet_truncated(
    imagenet_dataset_train,
    imagenet_dataset_test,
    train_bs,
    test_bs,
    dataidxs=None,
    net_dataidx_map=None,
):
    """
    imagenet_dataset_train, imagenet_dataset_test should be ImageNet or ImageNet_hdf5
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
    Used for generating distributed dataloader for
    accelerating centralized training
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
