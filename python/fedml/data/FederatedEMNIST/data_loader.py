import logging
import os

import h5py
import numpy as np
import torch
import torch.utils.data as data

client_ids_train = None
client_ids_test = None
DEFAULT_TRAIN_CLIENTS_NUM = 3400
DEFAULT_TEST_CLIENTS_NUM = 3400
DEFAULT_BATCH_SIZE = 20
DEFAULT_TRAIN_FILE = "fed_emnist_train.h5"
DEFAULT_TEST_FILE = "fed_emnist_test.h5"

# group name defined by tff in h5 file
_EXAMPLE = "examples"
_IMGAE = "pixels"
_LABEL = "label"


def get_dataloader(dataset, data_dir, train_bs, test_bs, client_idx=None):

    train_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TRAIN_FILE), "r")
    test_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TEST_FILE), "r")
    train_x = []
    test_x = []
    train_y = []
    test_y = []

    # load data
    if client_idx is None:
        # get ids of all clients
        train_ids = client_ids_train
        test_ids = client_ids_test
    else:
        # get ids of single client
        train_ids = [client_ids_train[client_idx]]
        test_ids = [client_ids_test[client_idx]]

    # load data in numpy format from h5 file
    train_x = np.vstack(
        [train_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in train_ids]
    )
    train_y = np.vstack(
        [train_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in train_ids]
    ).squeeze()
    test_x = np.vstack(
        [test_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in test_ids]
    )
    test_y = np.vstack(
        [test_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in test_ids]
    ).squeeze()

    # dataloader
    train_ds = data.TensorDataset(
        torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long)
    )
    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True
    )

    test_ds = data.TensorDataset(
        torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long)
    )
    test_dl = data.DataLoader(
        dataset=test_ds, batch_size=test_bs, shuffle=True, drop_last=False
    )

    train_h5.close()
    test_h5.close()
    return train_dl, test_dl


def load_partition_data_distributed_federated_emnist(
    process_id, dataset, data_dir, batch_size=DEFAULT_BATCH_SIZE
):

    if process_id == 0:
        # get global dataset
        train_data_global, test_data_global = get_dataloader(
            dataset, data_dir, batch_size, batch_size, process_id - 1
        )
        train_data_num = len(train_data_global)
        # logging.info("train_dl_global number = " + str(train_data_num))
        # logging.info("test_dl_global number = " + str(test_data_num))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
        test_file_path = os.path.join(data_dir, DEFAULT_TEST_FILE)
        with h5py.File(train_file_path, "r") as train_h5, h5py.File(
            test_file_path, "r"
        ) as test_h5:
            global client_ids_train, client_ids_test
            client_ids_train = list(train_h5[_EXAMPLE].keys())
            client_ids_test = list(test_h5[_EXAMPLE].keys())
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, process_id - 1
        )
        train_data_num = local_data_num = len(train_data_local)
        train_data_global = None
        test_data_global = None

    # class number
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    with h5py.File(train_file_path, "r") as train_h5:
        class_num = len(
            np.unique(
                [
                    train_h5[_EXAMPLE][client_ids_train[idx]][_LABEL][0]
                    for idx in range(DEFAULT_TRAIN_CLIENTS_NUM)
                ]
            )
        )
        logging.info("class_num = %d" % class_num)

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


def load_partition_data_federated_emnist(
    dataset, data_dir, batch_size=DEFAULT_BATCH_SIZE
):

    # client ids
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    test_file_path = os.path.join(data_dir, DEFAULT_TEST_FILE)
    with h5py.File(train_file_path, "r") as train_h5, h5py.File(
        test_file_path, "r"
    ) as test_h5:
        global client_ids_train, client_ids_test
        client_ids_train = list(train_h5[_EXAMPLE].keys())
        client_ids_test = list(test_h5[_EXAMPLE].keys())

    # local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(DEFAULT_TRAIN_CLIENTS_NUM):
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, client_idx
        )
        local_data_num = len(train_data_local) + len(test_data_local)
        data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
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

    # class number
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    with h5py.File(train_file_path, "r") as train_h5:
        class_num = len(
            np.unique(
                [
                    train_h5[_EXAMPLE][client_ids_train[idx]][_LABEL][0]
                    for idx in range(DEFAULT_TRAIN_CLIENTS_NUM)
                ]
            )
        )
        logging.info("class_num = %d" % class_num)

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
