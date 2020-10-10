import logging
import random

import h5py
import numpy as np
import torch
import torch.utils.data as data

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

client_map = None
DEFAULT_CLIENT_NUMBER = 3400
DEFAULT_BATCH_SIZE = 20
train_file_path = '../../../data/FederatedEMNIST/emnist_train.h5'
test_file_path = '../../../data/FederatedEMNIST/emnist_test.h5'


def get_client_map(client_id=None, client_num=None):
    global client_map
    if client_map == None:
        random.shuffle(client_id)
        client_map = {k: [client_id[i] for i in range(k, len(client_id), client_num)] for k in range(client_num)}
    return client_map


def get_dataloader(dataset, data_dir, train_bs, test_bs, client_idx=None):
    train_h5 = h5py.File(train_file_path, 'r')
    test_h5 = h5py.File(test_file_path, 'r')
    train_x, train_y, train_id = train_h5['pixels'], train_h5['label'], train_h5['id']
    train_y = train_y[:].astype(np.long)
    test_x, test_y, test_id = test_h5['pixels'], test_h5['label'], test_h5['id']
    test_y = test_y[:].astype(np.long)

    if client_idx is None:
        train_ds = data.TensorDataset(torch.tensor(train_x[:, :]), torch.tensor(train_y[:]))
        test_ds = data.TensorDataset(torch.tensor(test_x[:, :]), torch.tensor(test_y[:]))
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=True, drop_last=False)

    else:
        client_ids = get_client_map()[client_idx]
        train_h5_idx = np.array([], dtype=int)
        for client_id in client_ids:
            train_h5_idx = np.concatenate((train_h5_idx, np.argwhere(train_id[()] == client_id)[:, 0]))
        train_h5_idx.sort()
        train_ds = data.TensorDataset(torch.tensor(train_x[train_h5_idx, :]), torch.tensor(train_y[train_h5_idx]))
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)

        test_h5_idx = np.array([], dtype=int)
        for client_id in client_ids:
            test_h5_idx = np.concatenate((test_h5_idx, np.argwhere(test_id[()] == client_id)[:, 0]))
        test_h5_idx.sort()
        test_ds = data.TensorDataset(torch.tensor(test_x[test_h5_idx, :]), torch.tensor(test_y[test_h5_idx]))
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=True, drop_last=False)

    train_h5.close()
    test_h5.close()
    return train_dl, test_dl


def load_partition_data_distributed_federated_emnist(process_id, dataset, data_dir, client_number=None,
                                                     batch_size=DEFAULT_BATCH_SIZE):
    if client_number is None:
        client_number = DEFAULT_CLIENT_NUMBER

    train_h5 = h5py.File(train_file_path, 'r')
    class_num = len(np.unique(train_h5['label'][()]))
    logging.info("class_num = %d" % class_num)
    train_h5.close()

    # get global dataset
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size, process_id - 1)
        train_data_num = len(train_data_global)
#        test_data_num = len(test_data_global)
        # logging.info("train_dl_global number = " + str(train_data_num))
        # logging.info("test_dl_global number = " + str(test_data_num))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        train_h5 = h5py.File(train_file_path, 'r')
        get_client_map(train_h5['id'].value, client_number)
        train_h5.close()
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, process_id - 1)
        train_data_num = local_data_num = len(train_data_local)
        # logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        train_data_global = None
        test_data_global = None
    return client_number, train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_federated_emnist(dataset, data_dir, client_number=None, batch_size=DEFAULT_BATCH_SIZE):
    if client_number is None:
        client_number = DEFAULT_CLIENT_NUMBER

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    train_data_num = len(train_data_global)
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_h5 = h5py.File(train_file_path, 'r')
    get_client_map(np.unique(train_h5['id'][()]), client_number)
    class_num = len(np.unique(train_h5['label'][()]))
    train_h5.close()

    for client_idx in range(client_number):
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, client_idx)
        local_data_num = len(train_data_local) + len(test_data_local)
        data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    return client_number, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def test_federated_emnist():
    '''
    this function checks the data from dataloader is the same as the data from tff API
    '''
    import tensorflow_federated as tff
    import tensorflow_datasets as tfds
    client_num = 300
    test_num = 10  # use 'test_num = client_num' to test on all generated client dataset

    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    client_train_ds = list(iter(tfds.as_numpy(emnist_train.create_tf_dataset_from_all_clients())))
    client_test_ds = list(iter(tfds.as_numpy(emnist_test.create_tf_dataset_from_all_clients())))

    _, _, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict, _ = load_partition_data_federated_emnist(
        None, None, client_num, 1)
    client_train_dl = list(iter(train_data_global))
    client_test_dl = list(iter(test_data_global))

    assert (len(client_train_ds) == len(client_train_dl))
    assert (len(client_test_ds) == len(client_test_dl))

    for idx in random.sample(range(client_num), test_num):
        train_local_dl = list(iter(train_data_local_dict[idx]))
        train_local_dl = {str(dl[0].numpy().squeeze()): dl[1].numpy().squeeze() for dl in train_local_dl}
        test_local_dl = list(iter(test_data_local_dict[idx]))
        test_local_dl = {str(dl[0].numpy().squeeze()): dl[1].numpy().squeeze() for dl in test_local_dl}
        for client_id in get_client_map()[idx]:
            train_local_ds = list(
                iter(tfds.as_numpy(emnist_train.create_tf_dataset_for_client(client_id.decode("utf-8")))))
            train_local_ds = [(str(ds['pixels']), ds['label']) for ds in train_local_ds]
            for ds in train_local_ds:
                assert (ds[0] in train_local_dl)
                assert (ds[1] == train_local_dl[ds[0]])

            test_local_ds = list(
                iter(tfds.as_numpy(emnist_test.create_tf_dataset_for_client(client_id.decode("utf-8")))))
            test_local_ds = [(str(ds['pixels']), ds['label']) for ds in test_local_ds]
            for ds in test_local_ds:
                assert (ds[0] in test_local_dl)
                assert (ds[1] == test_local_dl[ds[0]])

        logging.info("Test for dataset on client = %d passed." % idx)

    logging.info("Tests for dataset passed.")


if __name__ == "__main__":
    load_partition_data_federated_emnist(None, None, 300, 128)
