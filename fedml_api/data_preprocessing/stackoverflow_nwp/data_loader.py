import logging

import h5py
import torch
import torch.utils.data as data

from . import utils

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

client_ids_train = None
client_ids_test = None
DEFAULT_TRAIN_CLINETS_NUM = 342477
DEFAULT_TEST_CLIENTS_NUM = 204088
DEFAULT_BATCH_SIZE = 16
train_file_path = '../../../data/stackoverflow/datasets/stackoverflow_train.h5'
test_file_path = '../../../data/stackoverflow/datasets/stackoverflow_test.h5'
heldout_file_path = '../../../data/stackoverflow/datasets/stackoverflow_held_out.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_TOKENS = 'tokens'


def get_dataloader(dataset, data_dir, train_bs, test_bs, client_idx=None):
    train_h5 = h5py.File(train_file_path, 'r')
    test_h5 = h5py.File(test_file_path, 'r')
    train_ds = []
    test_ds = []
    
    if client_idx is None:
        for client_id in client_ids_train:
            raw_train = train_h5[_EXAMPLE][client_id][_TOKENS][()]
            raw_train = [x.decode('utf8') for x in raw_train]
            train_ds.extend(utils.preprocess(raw_train))
        for client_id in client_ids_test:
            raw_test = test_h5[_EXAMPLE][client_id][_TOKENS][()]
            raw_test = [x.decode('utf8') for x in raw_test]
            test_ds.extend(utils.preprocess(raw_test))
    else:
        client_id_train = client_ids_train[client_idx]

        raw_train = train_h5[_EXAMPLE][client_id_train][_TOKENS][()]
        raw_train = [x.decode('utf8') for x in raw_train]
        train_ds.extend(utils.preprocess(raw_train))

        if client_idx <= len(client_ids_test) - 1:
            client_id_test = client_ids_test[client_idx]
            raw_test = test_h5[_EXAMPLE][client_id_test][_TOKENS][()]
            raw_test = [x.decode('utf8') for x in raw_test]
            test_ds.extend(utils.preprocess(raw_test))

    train_x, train_y = utils.split(train_ds)
    train_ds = data.TensorDataset(torch.tensor(train_x[:, :]),
                                  torch.tensor(train_y[:]))
    train_dl = data.DataLoader(dataset=train_ds,
                               batch_size=train_bs,
                               shuffle=True,
                               drop_last=False)
    if len(test_ds) != 0:
        test_x, test_y = utils.split(test_ds)
        test_ds = data.TensorDataset(torch.tensor(test_x[:, :]),
                                     torch.tensor(test_y[:]))
        test_dl = data.DataLoader(dataset=test_ds,
                                  batch_size=test_bs,
                                  shuffle=True,
                                  drop_last=False)
    else:
        test_dl = None

    train_h5.close()
    test_h5.close()
    return train_dl, test_dl


def load_partition_data_distributed_federated_stackoverflow_nwp(
        process_id, dataset, data_dir, batch_size = DEFAULT_BATCH_SIZE):

    #client id list
    train_h5 = h5py.File(train_file_path, 'r')
    test_h5 = h5py.File(test_file_path, 'r')
    global client_ids_train, client_ids_test
    client_ids_train = list(train_h5[_EXAMPLE].keys())
    client_ids_test = list(test_h5[_EXAMPLE].keys())
    train_h5.close()
    test_h5.close()

    # get global dataset
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(
            dataset, data_dir, batch_size, batch_size, process_id - 1)
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
            dataset, data_dir, batch_size, batch_size, process_id - 1)
        train_data_num = local_data_num = len(train_data_local.dataset)
        logging.info("rank = %d, local_sample_number = %d" %
                     (process_id, local_data_num))
        train_data_global = None
        test_data_global = None
        
    VOCAB_LEN = len(utils.get_word_dict()) + 1
    return DEFAULT_TRAIN_CLINETS_NUM, train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, VOCAB_LEN


def load_partition_data_federated_stackoverflow_nwp(dataset, data_dir, batch_size = DEFAULT_BATCH_SIZE):
    logging.info("load_partition_data_federated_stackoverflow_nwp START")
    
    class ConcatDataset(torch.utils.data.Dataset):
        def __init__(self, *datasets):
            self.datasets = datasets

        def __getitem__(self, i):
            return tuple(d[i] for d in self.datasets)

        def __len__(self):
            return min(len(d) for d in self.datasets)
        
    #client id list
    train_h5 = h5py.File(train_file_path, 'r')
    test_h5 = h5py.File(test_file_path, 'r')
    global client_ids_train, client_ids_test
    client_ids_train = list(train_h5[_EXAMPLE].keys())
    client_ids_test = list(test_h5[_EXAMPLE].keys())
    train_h5.close()
    test_h5.close()
    
    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    
    for client_idx in range(DEFAULT_TRAIN_CLINETS_NUM):

        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, batch_size, client_idx)
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" %
                     (client_idx, local_data_num))
        logging.info(
            "client_idx = %d, batch_num_train_local = %d"
            % (client_idx, len(train_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    train_data_global = torch.utils.data.DataLoader(
                ConcatDataset(
                    list(dl.dataset for dl in list(train_data_local_dict.values()))
                ),
                batch_size=batch_size, shuffle=True)
    train_data_num = len(train_data_global.dataset)
    
    test_data_global = torch.utils.data.DataLoader(
            ConcatDataset(
                list(dl.dataset for dl in list(test_data_local_dict.values()) if dl is not None)
            ),
            batch_size=batch_size, shuffle=True)
    test_data_num = len(test_data_global.dataset)

    VOCAB_LEN = len(utils.get_word_dict()) + 1
    return DEFAULT_TRAIN_CLINETS_NUM, train_data_num, test_data_num, train_data_global, test_data_global, \
        data_local_num_dict, train_data_local_dict, test_data_local_dict, VOCAB_LEN


if __name__ == "__main__":
    #load_partition_data_federated_stackoverflow(None, None, 100, 128)
    train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, VOCAB_LEN = load_partition_data_distributed_federated_stackoverflow_nwp(
        2, None, None, 128)
    # print(load_partition_data_federated_stackoverflow_nwp(None, None))
    print(train_data_local, test_data_local)
    print(VOCAB_LEN)