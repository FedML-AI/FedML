import logging

import torch.utils.data as data
import numpy as np

from torchvision import transforms
from fedml_api.data_preprocessing.coco.transforms import Normalize, ToTensor, FixedResize
from fedml_api.data_preprocessing.coco.datasets import CocoDataset
from fedml_core.non_iid_partition.noniid_partition import record_data_stats

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _data_transforms_coco():
    COCO_MEAN = (0.485, 0.456, 0.406)
    COCO_STD = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        FixedResize(513),
        Normalize(mean=COCO_MEAN, std=COCO_STD),
        ToTensor()
    ])

    return transform, transform


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_coco(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_coco_test(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_coco(datadir, train_bs, test_bs, dataidxs=None):
    transform_train, transform_test = _data_transforms_coco()

    train_ds = CocoDataset(datadir,
                           split='train',
                           transform=transform_train,
                           download_dataset=False,
                           dataidxs=dataidxs)
    test_ds = CocoDataset(datadir,
                          split='val',
                          transform=transform_test,
                          download_dataset=False)

    print(len(train_ds))

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl, train_ds.num_classes


def get_dataloader_coco_test(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    transform_train, transform_test = _data_transforms_coco()

    train_ds = CocoDataset(datadir,
                           split='train',
                           transform=transform_train,
                           download_dataset=False,
                           dataidxs=dataidxs_train)
    test_ds = CocoDataset(datadir,
                          split='val',
                          transform=transform_test,
                          download_dataset=False,
                          dataidxs=dataidxs_test)

    print(len(train_ds))

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl, train_ds.num_classes


def load_coco_data(datadir):
    transform_train, transform_test = _data_transforms_coco()

    train_ds = CocoDataset(datadir, split='train', transform=transform_train, download_dataset=False)
    test_ds = CocoDataset(datadir, split='val', transform=transform_test, download_dataset=False)

    return train_ds.img_ids, train_ds.target, train_ds.cat_ids, test_ds.img_ids, test_ds.target, test_ds.cat_ids


# Get a partition map for each client
def partition_data(datadir, partition, n_nets, alpha):
    traindata_cls_counts = None
    net_dataidx_map = None
    logging.info("*********partition data***************")
    train_images, train_targets, train_cat_ids, _, __, ___ = load_coco_data(datadir)
    n_train = len(train_images)  # Number of training samples

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)  # As many splits as n_nets = number of clients
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    # non-iid data distribution
    # TODO: Add custom non-iid distribution option - hetero-fix
    elif partition == "hetero":
        min_size = 0
        categories = train_cat_ids  # category names
        N = n_train  # Number of labels/training samples
        net_dataidx_map = {}

        while min_size < 10:
            # Create a list of empty lists for clients
            idx_batch = [[] for _1 in range(n_nets)]

            # note: one image may have multiple categories.
            for c, cat in enumerate(categories):
                # print(c, cat)
                if c > 0:
                    idx_k = np.asarray([np.any(train_targets[i] == cat) and not np.any(
                        np.in1d(train_targets[i], categories[:c])) for i in
                                        range(len(train_targets))])
                else:
                    idx_k = np.asarray(
                        [np.any(train_targets[i] == cat) for i in range(len(train_targets))])

                idx_k = np.where(idx_k)[0]  # Get the indices of images that have category = c
                np.random.shuffle(idx_k)

                # alpha, parameter for Dirichlet dist, vector containing positive concentration parameters (larger
                # the value more even the distribution)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))

                # Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])

                # Normalize across all samples
                proportions = proportions / proportions.sum()

                # eg. For 10 clients, 15 samples -> [0,0,2,2,2,2,14,14,14] -> 9 elements
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        traindata_cls_counts = record_data_stats(train_targets, net_dataidx_map)

    return net_dataidx_map, traindata_cls_counts


def load_partition_data_distributed_coco(process_id, dataset, data_dir, partition_method, partition_alpha,
                                         client_number, batch_size):
    net_dataidx_map, traindata_cls_counts = partition_data(data_dir,
                                                           partition_method,
                                                           client_number,
                                                           partition_alpha)
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local_dict = None
        test_data_local_dict = None
        data_local_num_dict = None
    else:
        # get local dataset
        client_id = process_id - 1
        dataidxs = net_dataidx_map[client_id]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        train_data_local, test_data_local, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                                      dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))

        data_local_num_dict = {client_id: local_data_num}
        train_data_local_dict = {client_id: train_data_local}
        test_data_local_dict = {client_id: test_data_local}
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, \
           test_data_local_dict, class_num


# Called from main_fedseg
def load_partition_data_coco(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    net_dataidx_map, traindata_cls_counts = partition_data(data_dir,
                                                           partition_method,
                                                           client_number,
                                                           partition_alpha)

    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # Global train and test data
    train_data_global, test_data_global, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size)
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
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        train_data_local, test_data_local, class_num = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                                      dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))

        # Store dataloaders for each client as they contain specific data
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
           train_data_local_dict, test_data_local_dict, class_num
