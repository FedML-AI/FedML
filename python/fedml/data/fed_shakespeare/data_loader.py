import logging
import os

import h5py
import torch
import torch.utils.data as data

from . import utils

client_ids_train = None
client_ids_test = None
DEFAULT_TRAIN_CLIENTS_NUM = 715
DEFAULT_TEST_CLIENTS_NUM = 715
DEFAULT_BATCH_SIZE = 4
DEFAULT_TRAIN_FILE = "shakespeare_train.h5"
DEFAULT_TEST_FILE = "shakespeare_test.h5"

# group name defined by tff in h5 file
_EXAMPLE = "examples"
_SNIPPETS = "snippets"


def get_dataloader(dataset, data_dir, train_bs, test_bs, client_idx=None):

    train_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TRAIN_FILE), "r")
    test_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TEST_FILE), "r")
    train_ds = []
    test_ds = []

    # load data
    if client_idx is None:
        # get ids of all clients
        train_ids = client_ids_train
        test_ids = client_ids_test
    else:
        # get ids of single client
        train_ids = [client_ids_train[client_idx]]
        test_ids = [client_ids_test[client_idx]]

    for client_id in train_ids:
        raw_train = train_h5[_EXAMPLE][client_id][_SNIPPETS][()]
        raw_train = [x.decode("utf8") for x in raw_train]
        train_ds.extend(utils.preprocess(raw_train))
    for client_id in test_ids:
        raw_test = test_h5[_EXAMPLE][client_id][_SNIPPETS][()]
        raw_test = [x.decode("utf8") for x in raw_test]
        test_ds.extend(utils.preprocess(raw_test))

    # split data
    train_x, train_y = utils.split(train_ds)
    test_x, test_y = utils.split(test_ds)
    train_ds = data.TensorDataset(torch.tensor(train_x[:, :]), torch.tensor(train_y[:]))
    test_ds = data.TensorDataset(torch.tensor(test_x[:, :]), torch.tensor(test_y[:]))
    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False
    )
    test_dl = data.DataLoader(
        dataset=test_ds, batch_size=test_bs, shuffle=True, drop_last=False
    )

    train_h5.close()
    test_h5.close()
    return train_dl, test_dl


def load_partition_data_distributed_federated_shakespeare(
    process_id, dataset, data_dir, batch_size=DEFAULT_BATCH_SIZE
):

    if process_id == 0:
        # get global dataset
        train_data_global, test_data_global = get_dataloader(
            dataset, data_dir, batch_size, batch_size, process_id - 1
        )
        train_data_num = len(train_data_global)
        test_data_num = len(test_data_global)
        logging.info("train_dl_global number = " + str(train_data_num))
        logging.info("test_dl_global number = " + str(test_data_num))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        # client id list
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
        train_data_num = local_data_num = len(train_data_local.dataset)
        logging.info(
            "rank = %d, local_sample_number = %d" % (process_id, local_data_num)
        )
        train_data_global = None
        test_data_global = None

    VOCAB_LEN = len(utils.get_word_dict()) + 1
    return (
        DEFAULT_TRAIN_CLIENTS_NUM,
        train_data_num,
        train_data_global,
        test_data_global,
        local_data_num,
        train_data_local,
        test_data_local,
        VOCAB_LEN,
    )


def load_partition_data_federated_shakespeare(
    dataset, data_dir, batch_size=DEFAULT_BATCH_SIZE
):

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
            "client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (client_idx, len(train_data_local), len(test_data_local))
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

    VOCAB_LEN = len(utils.get_word_dict()) + 1
    return (
        DEFAULT_TRAIN_CLIENTS_NUM,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        VOCAB_LEN,
    )

