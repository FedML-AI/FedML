import logging
import os

import h5py
import numpy as np
import torch
import torch.utils.data as data

from . import utils

client_ids_train = None
client_ids_test = None
DEFAULT_TRAIN_CLIENTS_NUM = 500
DEFAULT_TEST_CLIENTS_NUM = 100
DEFAULT_BATCH_SIZE = 20
DEFAULT_TRAIN_FILE = "fed_cifar100_train.h5"
DEFAULT_TEST_FILE = "fed_cifar100_test.h5"

# group name defined by tff in h5 file
_EXAMPLE = "examples"
_IMGAE = "image"
_LABEL = "label"


def get_dataloader(dataset, data_dir, train_bs, test_bs, client_idx=None):

    train_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TRAIN_FILE), "r")
    test_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TEST_FILE), "r")
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    # load data in numpy format from h5 file
    if client_idx is None:
        train_x = np.vstack(
            [
                train_h5[_EXAMPLE][client_id][_IMGAE][()]
                for client_id in client_ids_train
            ]
        )
        train_y = np.vstack(
            [
                train_h5[_EXAMPLE][client_id][_LABEL][()]
                for client_id in client_ids_train
            ]
        ).squeeze()
        test_x = np.vstack(
            [test_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in client_ids_test]
        )
        test_y = np.vstack(
            [test_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in client_ids_test]
        ).squeeze()
    else:
        client_id_train = client_ids_train[client_idx]
        train_x = np.vstack([train_h5[_EXAMPLE][client_id_train][_IMGAE][()]])
        train_y = np.vstack([train_h5[_EXAMPLE][client_id_train][_LABEL][()]]).squeeze()
        if client_idx <= len(client_ids_test) - 1:
            client_id_test = client_ids_test[client_idx]
            test_x = np.vstack([test_h5[_EXAMPLE][client_id_test][_IMGAE][()]])
            test_y = np.vstack(
                [test_h5[_EXAMPLE][client_id_test][_LABEL][()]]
            ).squeeze()

    # preprocess
    train_x = utils.preprocess_cifar_img(torch.tensor(train_x), train=True)
    train_y = torch.tensor(train_y)
    if len(test_x) != 0:
        test_x = utils.preprocess_cifar_img(torch.tensor(test_x), train=False)
        test_y = torch.tensor(test_y)

    # generate dataloader
    train_ds = data.TensorDataset(train_x, train_y)
    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False
    )

    if len(test_x) != 0:
        test_ds = data.TensorDataset(test_x, test_y)
        test_dl = data.DataLoader(
            dataset=test_ds, batch_size=test_bs, shuffle=True, drop_last=False
        )
    else:
        test_dl = None

    train_h5.close()
    test_h5.close()
    return train_dl, test_dl


def load_partition_data_distributed_federated_cifar100(
    process_id, dataset, data_dir, batch_size=DEFAULT_BATCH_SIZE
):

    class_num = 100

    if process_id == 0:
        # get global dataset
        train_data_global, test_data_global = get_dataloader(
            dataset, data_dir, batch_size, batch_size
        )
        train_data_num = len(train_data_global.dataset)
        test_data_num = len(test_data_global.dataset)
        logging.info("train_dl_global number = " + str(train_data_num))
        logging.info("test_dl_global number = " + str(test_data_num))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, process_id - 1
        )
        train_data_num = local_data_num = len(train_data_local.dataset)
        logging.info(
            "rank = %d, local_sample_number = %d" % (process_id, local_data_num)
        )
        train_data_global = None
        test_data_global = None
    return (
        DEFAULT_TRAIN_CLIENTS_NUM,
        train_data_num,
        train_data_global,
        test_data_global,
        local_data_num,
        train_data_local,
        test_data_local,
        class_num,
    )


def load_partition_data_federated_cifar100(
    dataset, data_dir, batch_size=DEFAULT_BATCH_SIZE
):

    class_num = 100

    # client id list
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    test_file_path = os.path.join(data_dir, DEFAULT_TEST_FILE)
    with h5py.File(train_file_path, "r") as train_h5, h5py.File(
        test_file_path, "r"
    ) as test_h5:
        global client_ids_train, client_ids_test
        client_ids_train = list(train_h5[_EXAMPLE].keys())
        client_ids_test = list(test_h5[_EXAMPLE].keys())

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(DEFAULT_TRAIN_CLIENTS_NUM):
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, client_idx
        )
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        logging.info(
            "client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num)
        )
        logging.info(
            "client_idx = %d, batch_num_train_local = %d"
            % (client_idx, len(train_data_local))
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    # global dataset
    train_data_global = data.DataLoader(
        data.ConcatDataset(
            list(dl.dataset for dl in list(train_data_local_dict.values()))
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    train_data_num = len(train_data_global.dataset)

    test_data_global = data.DataLoader(
        data.ConcatDataset(
            list(
                dl.dataset
                for dl in list(test_data_local_dict.values())
                if dl is not None
            )
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_data_num = len(test_data_global.dataset)

    return (
        DEFAULT_TRAIN_CLIENTS_NUM,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
