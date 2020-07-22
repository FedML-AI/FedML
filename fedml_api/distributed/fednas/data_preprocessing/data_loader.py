# we've changed to a faster solver
# from scipy.optimize import linear_sum_assignment
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from data_preprocessing.datasets import MNIST_truncated, CIFAR10_truncated, ImageFolderTruncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


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

    # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    # train_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: F.pad(
    #         Variable(x.unsqueeze(0), requires_grad=False),
    #         (4, 4, 4, 4), mode='reflect').data.squeeze()),
    #     transforms.ToPILImage(),
    #     transforms.RandomCrop(32),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    # train_transform.transforms.append(Cutout(16))
    #
    # # data prep for test set
    # valid_transform = transforms.Compose([transforms.ToTensor(), normalize])

    return train_transform, valid_transform

def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

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
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'cinic10':
        _train_dir = './data/cinic10/cinic-10-trainlarge/train'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        trainset = ImageFolderTruncated(_train_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                                                  transforms.Lambda(lambda x: F.pad(
                                                                                      Variable(x.unsqueeze(0),
                                                                                               requires_grad=False),
                                                                                      (4, 4, 4, 4),
                                                                                      mode='reflect').data.squeeze()),
                                                                                  transforms.ToPILImage(),
                                                                                  transforms.RandomCrop(32),
                                                                                  transforms.RandomHorizontalFlip(),
                                                                                  transforms.ToTensor(),
                                                                                  transforms.Normalize(mean=cinic_mean,
                                                                                                       std=cinic_std),
                                                                                  ]))
        y_train = trainset.get_train_labels
        n_train = y_train.shape[0]

    if partition == "homo":
        # total_num = int(n_train / 200)
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
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

    elif partition == "hetero-fbs":
        # in this part we conduct a experimental study on exploring the effect of increasing the number of batches
        # but the number of data points are approximately fixed for each batch
        # the arguments we need to use here are: `args.partition_step_size`, `args.local_points`, `args.partition_step`(step can be {0, 1, ..., args.partition_step_size - 1}).
        # Note that `args.partition` need to be fixed as "hetero-fbs" where fbs means fixed batch size
        net_dataidx_map = {}

        # stage 1st: homo partition
        idxs = np.random.permutation(n_train)
        total_num_batches = int(
            n_train / args.local_points)  # e.g. currently we have 180k, we want each local batch has 5k data points the `total_num_batches` becomes 36
        step_batch_idxs = np.array_split(idxs, args.partition_step_size)

        sub_partition_size = int(
            total_num_batches / args.partition_step_size)  # e.g. for `total_num_batches` at 36 and `args.partition_step_size` at 6, we have `sub_partition_size` at 6

        # stage 2nd: hetero partition
        n_batches = (args.partition_step + 1) * sub_partition_size
        min_size = 0
        K = 10

        # N = len(step_batch_idxs[args.step])
        baseline_indices = np.concatenate([step_batch_idxs[i] for i in range(args.partition_step + 1)])
        y_train = y_train[baseline_indices]
        N = y_train.shape[0]

        while min_size < 10:
            idx_batch = [[] for _ in range(n_batches)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_batches))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_batches) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        # we leave this to the end
        for j in range(n_batches):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
        return y_train, net_dataidx_map, traindata_cls_counts, baseline_indices

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts
    # return y_train, net_dataidx_map, traindata_cls_counts


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset in ('mnist', 'cifar10'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            transform_train, transform_test = _data_transforms_cifar10()

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    elif dataset == 'cinic10':
        # statistic for normalizing the dataset
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]

        cinic_directory = './data/cinic10'

        training_set = ImageFolderTruncated(cinic_directory + '/cinic-10-trainlarge/train',
                                            dataidxs=dataidxs,
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Lambda(lambda x: F.pad(
                                                                              Variable(x.unsqueeze(0),
                                                                                       requires_grad=False),
                                                                              (4, 4, 4, 4),
                                                                              mode='reflect').data.squeeze()),
                                                                          transforms.ToPILImage(),
                                                                          transforms.RandomCrop(32),
                                                                          transforms.RandomHorizontalFlip(),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize(mean=cinic_mean,
                                                                                               std=cinic_std),
                                                                          ]))
        train_dl = torch.utils.data.DataLoader(training_set, batch_size=train_bs, shuffle=True)
        logger.info(
            "Len of training set: {}, len of imgs in training set: {}, len of train dl: {}".format(len(training_set),
                                                                                                   len(
                                                                                                       training_set.imgs),
                                                                                                   len(train_dl)))

        test_dl = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(cinic_directory + '/cinic-10-trainlarge/test',
                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                           transforms.Normalize(mean=cinic_mean,
                                                                                                std=cinic_std)])),
            batch_size=test_bs, shuffle=False)

    return train_dl, test_dl
