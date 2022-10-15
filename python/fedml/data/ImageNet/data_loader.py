import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
import pickle

from .datasets import ImageNet
from .datasets import ImageNet_truncated
from .datasets_hdf5 import ImageNet_hdf5
from .datasets_hdf5 import ImageNet_truncated_hdf5

# from datasets import ImageNet
# from datasets import ImageNet_truncated
# from datasets_hdf5 import ImageNet_hdf5
# from datasets_hdf5 import ImageNet_truncated_hdf5



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
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
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




def partition_data(y_train, datadir, partition, n_nets, num_classes, alpha):
    np.random.seed(10)
    logging.info("*********partition data***************")
    # X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    # n_train = X_train.shape[0]
    # n_test = X_test.shape[0]
    n_train = y_train.shape[0]


    iters = 0
    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = num_classes
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
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
                iters += 1
                # print(f"iters: {iters}, ")

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return net_dataidx_map, traindata_cls_counts


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts

def get_dataloader_ImageNet_truncated(
    imagenet_dataset_train,
    imagenet_dataset_test,
    train_bs,
    test_bs,
    dataidxs=None,
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

    train_ds = dl_obj(
        imagenet_dataset_train,
        dataidxs,
    )
    test_ds = dl_obj(
        imagenet_dataset_test,
        dataidxs=None,
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
        train=True,
        transform=transform_train,
        download=False,
    )
    test_ds = dl_obj(
        datadir, train=False, transform=transform_test, download=False
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
        train=True,
        transform=transform_train,
        download=True,
    )
    test_ds = dl_obj(
        datadir,
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
            data_dir=data_dir, train=True, transform=transform_train
        )

        test_dataset = ImageNet(
            data_dir=data_dir, train=False, transform=transform_test
        )
    elif dataset == "ILSVRC2012_hdf5":
        train_dataset = ImageNet_hdf5(
            data_dir=data_dir, train=True, transform=transform_train
        )

        test_dataset = ImageNet_hdf5(
            data_dir=data_dir, train=False, transform=transform_test
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


def save_net_dataidx_map(
    dataset,
    data_dir,
    partition_method=None,
    partition_alpha=None,
    client_number=100,
    batch_size=10,
    net_dataidx_map_file=None,
):
    transform_train, transform_test = _data_transforms_ImageNet()

    if dataset == "ILSVRC2012":
        train_dataset = ImageNet(data_dir=data_dir, train=True, transform=transform_train)
        test_dataset = ImageNet(data_dir=data_dir, train=False, transform=transform_test)
    elif dataset == "ILSVRC2012_hdf5":
        train_dataset = ImageNet_hdf5(data_dir=data_dir, train=True, transform=transform_train)
        test_dataset = ImageNet_hdf5(data_dir=data_dir, train=False, transform=transform_test)

    class_num = 1000

    net_dataidx_map, traindata_cls_counts = partition_data(
                    train_dataset.targets, data_dir, partition_method, client_number, class_num, partition_alpha)
    with open(net_dataidx_map_file, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(net_dataidx_map, f)


def load_partition_data_ImageNet(
    dataset,
    data_dir,
    partition_method=None,
    partition_alpha=None,
    client_number=100,
    batch_size=10,
    net_dataidx_map_file=None,
):
    transform_train, transform_test = _data_transforms_ImageNet()

    if dataset == "ILSVRC2012":
        train_dataset = ImageNet(data_dir=data_dir, train=True, transform=transform_train)
        test_dataset = ImageNet(data_dir=data_dir, train=False, transform=transform_test)
    elif dataset == "ILSVRC2012_hdf5":
        train_dataset = ImageNet_hdf5(data_dir=data_dir, train=True, transform=transform_train)
        test_dataset = ImageNet_hdf5(data_dir=data_dir, train=False, transform=transform_test)

    class_num = 1000

    if net_dataidx_map_file is None:
        net_dataidx_map, traindata_cls_counts = partition_data(
                        train_dataset.targets, data_dir, partition_method, client_number, class_num, partition_alpha)
    else:
        with open(net_dataidx_map_file, 'rb') as f:
            net_dataidx_map = pickle.load(f)


    # print(net_dataidx_map)

    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    # print(f"train_data_num: {train_data_num}")
    # print(f"len(train_dataset.targets): {len(train_dataset.targets)}")
    train_data_num = len(train_dataset)
    # print(f"train_data_num: {train_data_num}")
    test_data_num = len(test_dataset)
    # class_num_dict = train_dataset.get_data_local_num_dict()

    train_data_global, test_data_global = get_dataloader_ImageNet_truncated(
        train_dataset,
        test_dataset,
        train_bs=batch_size,
        test_bs=batch_size,
        dataidxs=None,
    )

    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        # if client_number == 1000:
        #     dataidxs = client_idx
        #     data_local_num_dict = class_num_dict
        # elif client_number == 100:
        #     dataidxs = [client_idx * 10 + i for i in range(10)]
        #     data_local_num_dict[client_idx] = sum(
        #         class_num_dict[client_idx + i] for i in range(10)
        #     )
        # else:
        #     raise NotImplementedError("Not support other client_number for now!")

        dataidxs = net_dataidx_map[client_idx]
        train_data_local, test_data_local = get_dataloader_ImageNet_truncated(
            train_dataset,
            test_dataset,
            train_bs=batch_size,
            test_bs=batch_size,
            dataidxs=dataidxs,
        )
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num


        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
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
    partition_alpha = 0.1
    client_number = 1000
    # data_dir = '/home/datasets/imagenet/ILSVRC2012_dataset'
    data_dir = "/home/datasets/imagenet/imagenet_hdf5/imagenet-shuffled.hdf5"
    net_dataidx_map_file = f"imagenet_hdf5_c{client_number}_n1000_alpha{partition_alpha}.pth"

    save_net_dataidx_map(
        "ILSVRC2012_hdf5",
        data_dir,
        partition_method="hetero",
        partition_alpha=partition_alpha,
        client_number=client_number,
        batch_size=10,
        net_dataidx_map_file=net_dataidx_map_file
    )


    # (
    #     train_data_num,
    #     test_data_num,
    #     train_data_global,
    #     test_data_global,
    #     data_local_num_dict,
    #     train_data_local_dict,
    #     test_data_local_dict,
    #     class_num,
    # ) = load_partition_data_ImageNet(
    #     "ILSVRC2012_hdf5",
    #     data_dir,
    #     partition_method="hetero",
    #     partition_alpha=partition_alpha,
    #     client_number=client_number,
    #     batch_size=10,
    #     net_dataidx_map_file=net_dataidx_map_file
    # )

    # print(train_data_num, test_data_num, class_num)
    # print(data_local_num_dict)



    # i = 0
    # for data, label in train_data_global:
    #     print(data)
    #     print(label)
    #     i += 1
    #     if i > 5:
    #         break
    # print("=============================\n")

    # i = 0
    # for data, label in test_data_global:
    #     print(data)
    #     print(label)
    #     i += 1
    #     if i > 5:
    #         break
    # print("=============================\n")

    # for client_idx in range(client_number):
    #     i = 0
    #     for data, label in train_data_local_dict[client_idx]:
    #         print(data)
    #         print(label)
    #         i += 1
    #         if i > 5:
    #             break
