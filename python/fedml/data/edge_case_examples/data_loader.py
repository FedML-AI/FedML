import copy
import copy
import json
import logging
import os
import pickle
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import wget
from torch.autograd import Variable
from torchvision import datasets, transforms

from .datasets import (
    MNIST_truncated,
    EMNIST_truncated,
    CIFAR10_truncated,
    CIFAR10_Poisoned,
    CIFAR10NormalCase_truncated,
    EMNIST_NormalCase_truncated,
    ImageFolderTruncated,
)
import logging


def download_edgecase_data(data_cache_dir):
    file_path = data_cache_dir + "/edge_case_examples.zip"
    logging.info(file_path)
    URL = "http://pages.cs.wisc.edu/~hongyiwang/edge_case_attack/edge_case_examples.zip"
    # Download the file (if we haven't already)
    if not os.path.exists(file_path):
        wget.download(URL, out=file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_cache_dir)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(
        datadir, train=True, download=True, transform=transform
    )
    mnist_test_ds = MNIST_truncated(
        datadir, train=False, download=True, transform=transform
    )

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_emnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    emnist_train_ds = EMNIST_truncated(
        datadir, train=True, download=True, transform=transform
    )
    emnist_test_ds = EMNIST_truncated(
        datadir, train=False, download=True, transform=transform
    )

    X_train, y_train = emnist_train_ds.data, emnist_train_ds.target
    X_test, y_test = emnist_test_ds.data, emnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(
        datadir, train=True, download=True, transform=transform
    )
    cifar10_test_ds = CIFAR10_truncated(
        datadir, train=False, download=True, transform=transform
    )

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha, args):
    if dataset == "mnist":
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == "emnist":
        X_train, y_train, X_test, y_test = load_emnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset.lower() == "cifar10":
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)

        n_train = X_train.shape[0]

    elif dataset == "cinic10":
        _train_dir = "./data/cinic10/cinic-10-trainlarge/train"
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        trainset = ImageFolderTruncated(
            _train_dir,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: F.pad(
                            Variable(x.unsqueeze(0), requires_grad=False),
                            (4, 4, 4, 4),
                            mode="reflect",
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
        y_train = trainset.get_train_labels
        n_train = y_train.shape[0]
    elif dataset == "shakespeare":
        net_dataidx_map = {}
        with open(datadir[0]) as json_file:
            train_data = json.load(json_file)

        with open(datadir[1]) as json_file:
            test_data = json.load(json_file)

        for j in range(n_nets):
            client_user_name = train_data["users"][j]

            client_train_data = train_data["user_data"][client_user_name]["x"]
            num_samples_train = len(client_train_data)
            net_dataidx_map[j] = [
                i for i in range(num_samples_train)
            ]  # TODO: this is a dirty hack. needs modification
        return None, net_dataidx_map, None

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        while (min_size < 10) or (dataset == "mnist" and min_size < 100):
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

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        if dataset == "cifar10":
            if args.poison_type == "howto" or args.poison_type == "greencar-neo":
                green_car_indices = [
                    874,
                    49163,
                    34287,
                    21422,
                    48003,
                    47001,
                    48030,
                    22984,
                    37533,
                    41336,
                    3678,
                    37365,
                    19165,
                    34385,
                    41861,
                    39824,
                    561,
                    49588,
                    4528,
                    3378,
                    38658,
                    38735,
                    19500,
                    9744,
                    47026,
                    1605,
                    389,
                ] + [32941, 36005, 40138]
                # sanity_check_counter = 0
                for k, v in net_dataidx_map.items():
                    remaining_indices = [i for i in v if i not in green_car_indices]
                    # sanity_check_counter += len(remaining_indices)
                    net_dataidx_map[k] = remaining_indices

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return net_dataidx_map


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset in ("mnist", "emnist", "cifar10"):
        if dataset == "mnist":
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )

            transform_test = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        if dataset == "emnist":
            dl_obj = EMNIST_truncated

            transform_train = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )

            transform_test = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )

        elif dataset == "cifar10":
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            )
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: F.pad(
                            Variable(x.unsqueeze(0), requires_grad=False),
                            (4, 4, 4, 4),
                            mode="reflect",
                        ).data.squeeze()
                    ),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        train_ds = dl_obj(
            datadir,
            dataidxs=dataidxs,
            train=True,
            transform=transform_train,
            download=True,
        )
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl


def get_dataloader_normal_case(
    dataset,
    datadir,
    train_bs,
    test_bs,
    dataidxs=None,
    user_id=0,
    num_total_users=200,
    poison_type="southwest",
    ardis_dataset=None,
    attack_case="normal-case",
):
    if dataset in ("mnist", "emnist", "cifar10"):
        if dataset == "mnist":
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )

            transform_test = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        if dataset == "emnist":
            dl_obj = EMNIST_NormalCase_truncated

            transform_train = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )

            transform_test = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        elif dataset == "cifar10":
            dl_obj = CIFAR10NormalCase_truncated

            normalize = transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            )
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: F.pad(
                            Variable(x.unsqueeze(0), requires_grad=False),
                            (4, 4, 4, 4),
                            mode="reflect",
                        ).data.squeeze()
                    ),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        # this only supports cifar10 right now, please be super careful when calling it using other datasets
        train_ds = dl_obj(
            datadir,
            dataidxs=dataidxs,
            train=True,
            transform=transform_train,
            download=True,
            user_id=user_id,
            num_total_users=num_total_users,
            poison_type=poison_type,
            ardis_dataset_train=ardis_dataset,
            attack_case=attack_case,
        )

        test_ds = None  # dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl


def load_poisoned_dataset(args):
    use_cuda = not args.using_gpu and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    if args.dataset in ("mnist", "emnist"):
        if args.fraction < 1:
            fraction = args.fraction  # 0.1 #10
        else:
            fraction = int(args.fraction)

        with open(
            "poisoned_dataset_fraction_{}".format(fraction), "rb"
        ) as saved_data_file:
            poisoned_dataset = torch.load(saved_data_file)
        num_dps_poisoned_dataset = poisoned_dataset.data.shape[0]

        # prepare EMNIST dataset
        emnist_train_dataset = datasets.EMNIST(
            args.data_cache_dir,
            split="digits",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        emnist_test_dataset = datasets.EMNIST(
            args.data_cache_dir,
            split="digits",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        poisoned_train_loader = torch.utils.data.DataLoader(
            poisoned_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        vanilla_test_loader = torch.utils.data.DataLoader(
            emnist_test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )
        targetted_task_test_loader = torch.utils.data.DataLoader(
            emnist_test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )
        # clean_train_loader = torch.utils.data.DataLoader(emnist_train_dataset,
        #        batch_size=args.batch_size, shuffle=True, **kwargs)

        if args.poison_type == "ardis":
            # load ardis test set
            with open(
                "./../../../data/edge_case_examples/ARDIS/ardis_test_dataset.pt", "rb"
            ) as saved_data_file:
                ardis_test_dataset = torch.load(saved_data_file)

            targetted_task_test_loader = torch.utils.data.DataLoader(
                ardis_test_dataset,
                batch_size=args.test_batch_size,
                shuffle=False,
                **kwargs
            )

    elif args.dataset == "cifar10":
        if args.poison_type == "southwest":
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

            trainset = torchvision.datasets.CIFAR10(
                root=args.data_cache_dir,
                train=True,
                download=True,
                transform=transform_train,
            )

            poisoned_trainset = copy.deepcopy(trainset)

            if args.attack_case == "edge-case":
                with open(
                    args.data_cache_dir
                    + "/southwest_cifar10/southwest_images_new_train.pkl",
                    "rb",
                ) as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)

                with open(
                    args.data_cache_dir
                    + "/southwest_cifar10/southwest_images_new_test.pkl",
                    "rb",
                ) as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)
            elif (
                args.attack_case == "normal-case"
                or args.attack_case == "almost-edge-case"
            ):
                with open(
                    args.data_cache_dir
                    + "/southwest_cifar10/southwest_images_adv_p_percent_edge_case.pkl",
                    "rb",
                ) as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)

                with open(
                    args.data_cache_dir
                    + "/southwest_cifar10/southwest_images_p_percent_edge_case_test.pkl",
                    "rb",
                ) as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)
            else:
                raise NotImplementedError("Not Matched Attack Case ...")

                #
            logging.info(
                "OOD (Southwest Airline) train-data shape we collected: {}".format(
                    saved_southwest_dataset_train.shape
                )
            )
            # sampled_targets_array_train = 2 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as bird
            sampled_targets_array_train = 9 * np.ones(
                (saved_southwest_dataset_train.shape[0],), dtype=int
            )  # southwest airplane -> label as truck

            logging.info(
                "OOD (Southwest Airline) test-data shape we collected: {}".format(
                    saved_southwest_dataset_test.shape
                )
            )
            # sampled_targets_array_test = 2 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as bird
            sampled_targets_array_test = 9 * np.ones(
                (saved_southwest_dataset_test.shape[0],), dtype=int
            )  # southwest airplane -> label as truck

            # downsample the poisoned dataset #################
            if args.attack_case == "edge-case":
                num_sampled_poisoned_data_points = 100  # N
                samped_poisoned_data_indices = np.random.choice(
                    saved_southwest_dataset_train.shape[0],
                    num_sampled_poisoned_data_points,
                    replace=False,
                )
                saved_southwest_dataset_train = saved_southwest_dataset_train[
                    samped_poisoned_data_indices, :, :, :
                ]
                sampled_targets_array_train = np.array(sampled_targets_array_train)[
                    samped_poisoned_data_indices
                ]
                logging.info(
                    "!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(
                        num_sampled_poisoned_data_points
                    )
                )
            elif (
                args.attack_case == "normal-case"
                or args.attack_case == "almost-edge-case"
            ):
                num_sampled_poisoned_data_points = 100  # N
                samped_poisoned_data_indices = np.random.choice(
                    784, num_sampled_poisoned_data_points, replace=False
                )
            ######################################################

            # downsample the raw cifar10 dataset #################
            num_sampled_data_points = 400  # M
            samped_data_indices = np.random.choice(
                poisoned_trainset.data.shape[0], num_sampled_data_points, replace=False
            )
            poisoned_trainset.data = poisoned_trainset.data[
                samped_data_indices, :, :, :
            ]
            poisoned_trainset.targets = np.array(poisoned_trainset.targets)[
                samped_data_indices
            ]
            logging.info(
                "!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(
                    num_sampled_data_points
                )
            )
            # keep a copy of clean data
            clean_trainset = copy.deepcopy(poisoned_trainset)
            ########################################################

            poisoned_trainset.data = np.append(
                poisoned_trainset.data, saved_southwest_dataset_train, axis=0
            )
            poisoned_trainset.targets = np.append(
                poisoned_trainset.targets, sampled_targets_array_train, axis=0
            )

            logging.info("{}".format(poisoned_trainset.data.shape))
            logging.info("{}".format(poisoned_trainset.targets.shape))
            logging.info("{}".format(sum(poisoned_trainset.targets)))

            # poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            poisoned_train_loader = torch.utils.data.DataLoader(
                poisoned_trainset, batch_size=args.batch_size, shuffle=True
            )
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True
            )
            # clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)

            testset = torchvision.datasets.CIFAR10(
                root=args.data_cache_dir,
                train=False,
                download=True,
                transform=transform_test,
            )

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_southwest_dataset_test
            poisoned_testset.targets = sampled_targets_array_test

            # vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
            # targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
            vanilla_test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.test_batch_size, shuffle=False
            )
            targetted_task_test_loader = torch.utils.data.DataLoader(
                poisoned_testset, batch_size=args.test_batch_size, shuffle=False
            )

            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]

        elif args.poison_type == "southwest-da":

            normalize = transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            )
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: F.pad(
                            Variable(x.unsqueeze(0), requires_grad=False),
                            (4, 4, 4, 4),
                            mode="reflect",
                        ).data.squeeze()
                    ),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

            transform_poison = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: F.pad(
                            Variable(x.unsqueeze(0), requires_grad=False),
                            (4, 4, 4, 4),
                            mode="reflect",
                        ).data.squeeze()
                    ),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                    AddGaussianNoise(0.0, 0.05), # pylint: disable=E0602
                ]
            )
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])

            trainset = torchvision.datasets.CIFAR10(
                root=args.data_cache_dir,
                train=True,
                download=True,
                transform=transform_train,
            )

            with open(
                args.data_cache_dir
                + "/southwest_cifar10/southwest_images_new_train.pkl",
                "rb",
            ) as train_f:
                saved_southwest_dataset_train = pickle.load(train_f)

            with open(
                args.data_cache_dir
                + "/southwest_cifar10/southwest_images_new_test.pkl",
                "rb",
            ) as test_f:
                saved_southwest_dataset_test = pickle.load(test_f)

            #
            logging.info(
                "OOD (Southwest Airline) train-data shape we collected: {}".format(
                    saved_southwest_dataset_train.shape
                )
            )
            sampled_targets_array_train = 9 * np.ones(
                (saved_southwest_dataset_train.shape[0],), dtype=int
            )  # southwest airplane -> label as truck

            logging.info(
                "OOD (Southwest Airline) test-data shape we collected: {}".format(
                    saved_southwest_dataset_test.shape
                )
            )
            sampled_targets_array_test = 9 * np.ones(
                (saved_southwest_dataset_test.shape[0],), dtype=int
            )  # southwest airplane -> label as truck

            # downsample the poisoned dataset ###########################
            num_sampled_poisoned_data_points = 100  # N
            samped_poisoned_data_indices = np.random.choice(
                saved_southwest_dataset_train.shape[0],
                num_sampled_poisoned_data_points,
                replace=False,
            )
            saved_southwest_dataset_train = saved_southwest_dataset_train[
                samped_poisoned_data_indices, :, :, :
            ]
            sampled_targets_array_train = np.array(sampled_targets_array_train)[
                samped_poisoned_data_indices
            ]
            logging.info(
                "!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(
                    num_sampled_poisoned_data_points
                )
            )
            ###############################################################

            # downsample the raw cifar10 dataset #################
            num_sampled_data_points = 400  # M
            samped_data_indices = np.random.choice(
                trainset.data.shape[0], num_sampled_data_points, replace=False
            )
            tempt_poisoned_trainset = trainset.data[samped_data_indices, :, :, :]
            tempt_poisoned_targets = np.array(trainset.targets)[samped_data_indices]
            logging.info(
                "!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(
                    num_sampled_data_points
                )
            )
            ########################################################

            poisoned_trainset = CIFAR10_Poisoned(
                root=args.data_cache_dir,
                clean_indices=np.arange(tempt_poisoned_trainset.shape[0]),
                poisoned_indices=np.arange(
                    tempt_poisoned_trainset.shape[0],
                    tempt_poisoned_trainset.shape[0]
                    + saved_southwest_dataset_train.shape[0],
                ),
                train=True,
                download=True,
                transform_clean=transform_train,
                transform_poison=transform_poison,
            )
            # poisoned_trainset = CIFAR10_truncated(root='./data', dataidxs=None, train=True, transform=transform_train, download=True)
            clean_trainset = copy.deepcopy(poisoned_trainset)

            poisoned_trainset.data = np.append(
                tempt_poisoned_trainset, saved_southwest_dataset_train, axis=0
            )
            poisoned_trainset.target = np.append(
                tempt_poisoned_targets, sampled_targets_array_train, axis=0
            )

            logging.info("{}".format(poisoned_trainset.data.shape))
            logging.info("{}".format(poisoned_trainset.target.shape))

            poisoned_train_loader = torch.utils.data.DataLoader(
                poisoned_trainset, batch_size=args.batch_size, shuffle=True
            )
            # clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True
            )

            testset = torchvision.datasets.CIFAR10(
                root=args.data_cache_dir,
                train=False,
                download=True,
                transform=transform_test,
            )

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_southwest_dataset_test
            poisoned_testset.targets = sampled_targets_array_test

            vanilla_test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.test_batch_size, shuffle=False
            )
            targetted_task_test_loader = torch.utils.data.DataLoader(
                poisoned_testset, batch_size=args.test_batch_size, shuffle=False
            )

            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]

        elif args.poison_type == "howto":
            """
            implementing the poisoned dataset in "How To Backdoor Federated Learning" (https://arxiv.org/abs/1807.00459)
            """
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

            trainset = torchvision.datasets.CIFAR10(
                root=args.data_cache_dir,
                train=True,
                download=True,
                transform=transform_train,
            )

            poisoned_trainset = copy.deepcopy(trainset)

            ##########################################################################################################################
            sampled_indices_train = [
                874,
                49163,
                34287,
                21422,
                48003,
                47001,
                48030,
                22984,
                37533,
                41336,
                3678,
                37365,
                19165,
                34385,
                41861,
                39824,
                561,
                49588,
                4528,
                3378,
                38658,
                38735,
                19500,
                9744,
                47026,
                1605,
                389,
            ]
            sampled_indices_test = [32941, 36005, 40138]
            cifar10_whole_range = np.arange(trainset.data.shape[0])
            remaining_indices = [
                i
                for i in cifar10_whole_range
                if i not in sampled_indices_train + sampled_indices_test
            ]
            logging.info(
                "!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(
                    len(sampled_indices_train + sampled_indices_test)
                )
            )
            saved_greencar_dataset_train = trainset.data[sampled_indices_train, :, :, :]
            #########################################################################################################################

            # downsample the raw cifar10 dataset ####################################################################################
            num_sampled_data_points = 500 - len(sampled_indices_train)
            samped_data_indices = np.random.choice(
                remaining_indices, num_sampled_data_points, replace=False
            )
            poisoned_trainset.data = poisoned_trainset.data[
                samped_data_indices, :, :, :
            ]
            poisoned_trainset.targets = np.array(poisoned_trainset.targets)[
                samped_data_indices
            ]
            logging.info(
                "!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(
                    num_sampled_data_points
                )
            )
            clean_trainset = copy.deepcopy(poisoned_trainset)
            ##########################################################################################################################

            # we load the test since in the original paper they augment the
            with open(
                args.data_cache_dir
                + "/greencar_cifar10/green_car_transformed_test.pkl",
                "rb",
            ) as test_f:
                saved_greencar_dataset_test = pickle.load(test_f)

            #
            logging.info(
                "Backdoor (Green car) train-data shape we collected: {}".format(
                    saved_greencar_dataset_train.shape
                )
            )
            sampled_targets_array_train = 2 * np.ones(
                (saved_greencar_dataset_train.shape[0],), dtype=int
            )  # green car -> label as bird

            logging.info(
                "Backdoor (Green car) test-data shape we collected: {}".format(
                    saved_greencar_dataset_test.shape
                )
            )
            sampled_targets_array_test = 2 * np.ones(
                (saved_greencar_dataset_test.shape[0],), dtype=int
            )  # green car -> label as bird/

            poisoned_trainset.data = np.append(
                poisoned_trainset.data, saved_greencar_dataset_train, axis=0
            )
            poisoned_trainset.targets = np.append(
                poisoned_trainset.targets, sampled_targets_array_train, axis=0
            )

            logging.info(
                "Poisoned Trainset Shape: {}".format(poisoned_trainset.data.shape)
            )
            logging.info(
                "Poisoned Train Target Shape:{}".format(poisoned_trainset.targets.shape)
            )

            poisoned_train_loader = torch.utils.data.DataLoader(
                poisoned_trainset, batch_size=args.batch_size, shuffle=True
            )
            # clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True
            )

            testset = torchvision.datasets.CIFAR10(
                root=args.data_cache_dir,
                train=False,
                download=True,
                transform=transform_test,
            )

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_greencar_dataset_test
            poisoned_testset.targets = sampled_targets_array_test

            vanilla_test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.test_batch_size, shuffle=False
            )
            targetted_task_test_loader = torch.utils.data.DataLoader(
                poisoned_testset, batch_size=args.test_batch_size, shuffle=False
            )
            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]

        elif args.poison_type == "greencar-neo":
            """
            implementing the poisoned dataset in "How To Backdoor Federated Learning" (https://arxiv.org/abs/1807.00459)
            """
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

            trainset = torchvision.datasets.CIFAR10(
                root=args.data_cache_dir,
                train=True,
                download=True,
                transform=transform_train,
            )

            poisoned_trainset = copy.deepcopy(trainset)

            with open(
                args.data_cache_dir + "/greencar_cifar10/new_green_cars_train.pkl",
                "rb",
            ) as train_f:
                saved_new_green_cars_train = pickle.load(train_f)

            with open(
                args.data_cache_dir + "/greencar_cifar10/new_green_cars_test.pkl",
                "rb",
            ) as test_f:
                saved_new_green_cars_test = pickle.load(test_f)

            # we use the green cars in original cifar-10 and new collected green cars
            ##########################################################################################################################
            num_sampled_poisoned_data_points = 100  # N
            sampled_indices_green_car = [
                874,
                49163,
                34287,
                21422,
                48003,
                47001,
                48030,
                22984,
                37533,
                41336,
                3678,
                37365,
                19165,
                34385,
                41861,
                39824,
                561,
                49588,
                4528,
                3378,
                38658,
                38735,
                19500,
                9744,
                47026,
                1605,
                389,
            ] + [32941, 36005, 40138]
            cifar10_whole_range = np.arange(trainset.data.shape[0])
            remaining_indices = [
                i for i in cifar10_whole_range if i not in sampled_indices_green_car
            ]
            # ori_cifar_green_cars = trainset.data[sampled_indices_green_car, :, :, :]

            samped_poisoned_data_indices = np.random.choice(
                saved_new_green_cars_train.shape[0],
                # num_sampled_poisoned_data_points-len(sampled_indices_green_car),
                num_sampled_poisoned_data_points,
                replace=False,
            )
            saved_new_green_cars_train = saved_new_green_cars_train[
                samped_poisoned_data_indices, :, :, :
            ]

            # saved_greencar_dataset_train = np.append(ori_cifar_green_cars, saved_new_green_cars_train, axis=0)
            saved_greencar_dataset_train = saved_new_green_cars_train
            logging.info(
                "!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(
                    saved_greencar_dataset_train.shape[0]
                )
            )
            #########################################################################################################################

            # downsample the raw cifar10 dataset ####################################################################################
            num_sampled_data_points = 400
            samped_data_indices = np.random.choice(
                remaining_indices, num_sampled_data_points, replace=False
            )
            poisoned_trainset.data = poisoned_trainset.data[
                samped_data_indices, :, :, :
            ]
            poisoned_trainset.targets = np.array(poisoned_trainset.targets)[
                samped_data_indices
            ]
            logging.info(
                "!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(
                    num_sampled_data_points
                )
            )
            clean_trainset = copy.deepcopy(poisoned_trainset)
            ##########################################################################################################################

            #
            logging.info(
                "Backdoor (Green car) train-data shape we collected: {}".format(
                    saved_greencar_dataset_train.shape
                )
            )
            sampled_targets_array_train = 2 * np.ones(
                (saved_greencar_dataset_train.shape[0],), dtype=int
            )  # green car -> label as bird

            logging.info(
                "Backdoor (Green car) test-data shape we collected: {}".format(
                    saved_new_green_cars_test.shape
                )
            )
            sampled_targets_array_test = 2 * np.ones(
                (saved_new_green_cars_test.shape[0],), dtype=int
            )  # green car -> label as bird/

            poisoned_trainset.data = np.append(
                poisoned_trainset.data, saved_greencar_dataset_train, axis=0
            )
            poisoned_trainset.targets = np.append(
                poisoned_trainset.targets, sampled_targets_array_train, axis=0
            )

            logging.info(
                "Poisoned Trainset Shape: {}".format(poisoned_trainset.data.shape)
            )
            logging.info(
                "Poisoned Train Target Shape:{}".format(poisoned_trainset.targets.shape)
            )

            poisoned_train_loader = torch.utils.data.DataLoader(
                poisoned_trainset, batch_size=args.batch_size, shuffle=True
            )
            # clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True
            )

            testset = torchvision.datasets.CIFAR10(
                root=args.data_cache_dir,
                train=False,
                download=True,
                transform=transform_test,
            )

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_new_green_cars_test
            poisoned_testset.targets = sampled_targets_array_test

            vanilla_test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.test_batch_size, shuffle=False
            )
            targetted_task_test_loader = torch.utils.data.DataLoader(
                poisoned_testset, batch_size=args.test_batch_size, shuffle=False
            )
            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]

    return poisoned_train_loader, targetted_task_test_loader, num_dps_poisoned_dataset
