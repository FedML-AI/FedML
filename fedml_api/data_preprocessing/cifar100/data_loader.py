import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from fedml_api.data_preprocessing.cifar10.datasets import CIFAR10_truncated, CIFAR100_truncated, ImageFolderTruncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


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

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def _data_transforms_cifar100():
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def _data_transforms_cinic10():
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    # Transformer for train set: random crops and horizontal flip
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Lambda(
                                              lambda x: F.pad(x.unsqueeze(0),
                                                              (4, 4, 4, 4),
                                                              mode='reflect').data.squeeze()),
                                          transforms.ToPILImage(),
                                          transforms.RandomCrop(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=cinic_mean,
                                                               std=cinic_std),
                                          ])

    # Transformer for test set
    valid_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Lambda(
                                              lambda x: F.pad(x.unsqueeze(0),
                                                              (4, 4, 4, 4),
                                                              mode='reflect').data.squeeze()),
                                          transforms.ToPILImage(),
                                          transforms.RandomCrop(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=cinic_mean,
                                                               std=cinic_std),
                                          ])
    return train_transform, valid_transform


def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    train_transform, test_transform = _data_transforms_cifar100()

    cifar10_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def _data_transforms_imagenet():
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # Transformer for train set: random crops and horizontal flip
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.ToTensor(),
        normalize])

    # Transformer for test set
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, valid_transform


def load_imagenet_data(datadir):
    train_transform, test_transform = _data_transforms_imagenet()

    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'val')

    imagenet_train_ds = ImageFolderTruncated(traindir, transform=train_transform)
    imagenet_test_ds = ImageFolderTruncated(valdir, transform=test_transform)

    X_train, y_train = imagenet_train_ds.imgs, imagenet_train_ds.targets
    X_test, y_test = imagenet_test_ds.imgs, imagenet_test_ds.targets

    return (X_train, y_train, X_test, y_test)


def load_cinic10_data(datadir):
    _train_dir = datadir + str('/train')
    logging.info("_train_dir = " + str(_train_dir))
    _test_dir = datadir + str('/test')
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    trainset = ImageFolderTruncated(_train_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                                              transforms.Lambda(
                                                                                  lambda x: F.pad(x.unsqueeze(0),
                                                                                                  (4, 4, 4, 4),
                                                                                                  mode='reflect').data.squeeze()),
                                                                              transforms.ToPILImage(),
                                                                              transforms.RandomCrop(32),
                                                                              transforms.RandomHorizontalFlip(),
                                                                              transforms.ToTensor(),
                                                                              transforms.Normalize(mean=cinic_mean,
                                                                                                   std=cinic_std),
                                                                              ]))

    testset = ImageFolderTruncated(_test_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Lambda(
                                                                                lambda x: F.pad(x.unsqueeze(0),
                                                                                                (4, 4, 4, 4),
                                                                                                mode='reflect').data.squeeze()),
                                                                            transforms.ToPILImage(),
                                                                            transforms.RandomCrop(32),
                                                                            transforms.RandomHorizontalFlip(),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize(mean=cinic_mean,
                                                                                                 std=cinic_std),
                                                                            ]))
    X_train, y_train = trainset.imgs, trainset.targets
    X_test, y_test = testset.imgs, testset.targets
    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_nets, alpha, args):
    if dataset == 'cifar10':
        logging.info("************************1")
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

    elif dataset == 'cifar100':
        logging.info("************************2")
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

    elif dataset == 'cinic10':
        logging.info("************************3")
        X_train, y_train, X_test, y_test = load_cinic10_data(datadir)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        n_train = len(X_train)
        n_test = len(X_test)

    elif dataset == 'imagenet':
        logging.info("************************4")
        X_train, y_train, X_test, y_test = load_imagenet_data(datadir)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        n_train = len(X_train)
        n_test = len(X_test)

    else:
        X_train, y_train, X_test, y_test = load_imagenet_data(datadir)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        n_train = len(X_train)
        n_test = len(X_test)
        logging.info("n_train = " + str(n_train))
        logging.info("n_test = " + str(n_test))

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

        total_num_test = n_test
        idxs_test = np.random.permutation(total_num_test)
        batch_idxs_test = np.array_split(idxs_test, n_nets)
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        if dataset == 'cifar10':
            K = 10
        elif dataset == 'cinic10':
            K = 10
        elif dataset == 'cifar100':
            K = 100
        elif dataset == 'imagenet':
            K = 1000
        else:
            K = 1000
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
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        # every worker has the same number of test samples
        total_num_test = n_test
        idxs_test = np.random.permutation(total_num_test)
        batch_idxs_test = np.array_split(idxs_test, n_nets)
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_nets)}
    elif partition == "hetero-fix":
        if dataset == 'cifar10':
            dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
        elif dataset == 'cinic10':
            dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CINIC10/net_dataidx_map.txt'
        elif dataset == 'cifar100':
            dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR100/net_dataidx_map.txt'
        else:
            dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)
    else:
        # test
        total_num = int(n_train / 200)
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

        total_num_test = int(n_test / 20)
        idxs_test = np.random.permutation(total_num_test)
        batch_idxs_test = np.array_split(idxs_test, n_nets)
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_nets)}

    if partition == "hetero-fix":
        if dataset == 'cifar10':
            distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        elif dataset == 'cinic10':
            distribution_file_path = './data_preprocessing/non-iid-distribution/CINIC10/distribution.txt'
        elif dataset == 'cifar100':
            distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR100/distribution.txt'
        else:
            distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset == 'cifar10':
        logging.info("#########################1")
        return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs)
    elif dataset == 'cifar100':
        logging.info("#########################2")
        return get_dataloader_CIFAR100(datadir, train_bs, test_bs, dataidxs)
    elif dataset == 'cinic10':
        logging.info("#########################3")
        return get_dataloader_cinic10(datadir, train_bs, test_bs, dataidxs)
    elif dataset == 'imagenet':
        logging.info("#########################4")
        return get_dataloader_ImageNet(datadir, train_bs, test_bs, dataidxs)
    else:
        return get_dataloader_ImageNet(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    if dataset == 'cifar10':
        return get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)
    elif dataset == 'cifar100':
        return get_dataloader_test_CIFAR100(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)
    elif dataset == 'cinic10':
        return get_dataloader_test_cinic10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)
    elif dataset == 'imagenet':
        return get_dataloader_test_ImageNet(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)
    else:
        return get_dataloader_test_ImageNet(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_CIFAR100(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = CIFAR100_truncated

    transform_train, transform_test = _data_transforms_cifar100()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_CIFAR100(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR100_truncated

    transform_train, transform_test = _data_transforms_cifar100()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_cinic10(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_cinic10()

    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'test')

    train_ds = dl_obj(traindir, dataidxs=dataidxs, transform=transform_train)
    test_ds = dl_obj(valdir, transform=transform_train)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_cinic10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_cinic10()

    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'test')

    train_ds = dl_obj(traindir, dataidxs=dataidxs_train, transform=transform_train)
    test_ds = dl_obj(valdir, dataidxs=dataidxs_test, transform=transform_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_ImageNet(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_imagenet()

    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'val')

    train_ds = dl_obj(traindir, dataidxs=dataidxs, transform=transform_train)
    test_ds = dl_obj(valdir, transform=transform_train)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_ImageNet(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_imagenet()

    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'val')

    train_ds = dl_obj(traindir, dataidxs=dataidxs_train, transform=transform_train)
    test_ds = dl_obj(valdir, dataidxs=dataidxs_test, transform=transform_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl
