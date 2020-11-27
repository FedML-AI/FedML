import logging

import h5py
import numpy as np
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
DEFAULT_BATCH_SIZE = 100
train_file_path = '../../../data/stackoverflow/datasets/stackoverflow_train.h5'
test_file_path = '../../../data/stackoverflow/datasets/stackoverflow_test.h5'
heldout_file_path = '../../../data/stackoverflow/datasets/stackoverflow_held_out.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_TOKENS = 'tokens'
_TITLE = 'title'
_TAGS = 'tags'


def get_dataloader(dataset, data_dir, train_bs, test_bs, client_idx=None):
    train_h5 = h5py.File(train_file_path, 'r')
    test_h5 = h5py.File(test_file_path, 'r')
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    
    if client_idx is None:
        for client_id in client_ids_train:
            raw_tokens = train_h5[_EXAMPLE][client_id][_TOKENS][()]
            raw_tokens = [x.decode('utf8') for x in raw_tokens]
            raw_title = train_h5[_EXAMPLE][client_id][_TITLE][()]
            raw_title = [x.decode('utf8') for x in raw_title]
            raw_x = [' '.join(pair) for pair in zip(raw_tokens, raw_title)]
            raw_y = [
                x.decode('utf8') for x in train_h5[_EXAMPLE][client_id][_TAGS][()]
            ]
            train_x.extend(utils.preprocess_inputs(raw_x))
            train_y.extend(utils.preprocess_targets(raw_y))
        for client_id in client_ids_test:
            raw_tokens_test = test_h5[_EXAMPLE][client_id][_TOKENS][()]
            raw_tokens_test = [x.decode('utf8') for x in raw_tokens_test]
            raw_title_test = test_h5[_EXAMPLE][client_id][_TITLE][()]
            raw_title_test = [x.decode('utf8') for x in raw_title_test]
            raw_x_test = [
                ' '.join(pair) for pair in zip(raw_tokens_test, raw_title_test)
            ]
            raw_y_test = [
                x.decode('utf8') for x in test_h5[_EXAMPLE][client_id][_TAGS][()]
            ]
            test_x.extend(utils.preprocess_inputs(raw_x_test))
            test_y.extend(utils.preprocess_targets(raw_y_test))
    else:
        client_id_train = client_ids_train[client_idx]
        raw_tokens = train_h5[_EXAMPLE][client_id_train][_TOKENS][()]
        raw_tokens = [x.decode('utf8') for x in raw_tokens]
        raw_title = train_h5[_EXAMPLE][client_id_train][_TITLE][()]
        raw_title = [x.decode('utf8') for x in raw_title]
        raw_x = [' '.join(pair) for pair in zip(raw_tokens, raw_title)]
        raw_y = [
            x.decode('utf8') for x in train_h5[_EXAMPLE][client_id_train][_TAGS][()]
        ]
        train_x.extend(utils.preprocess_inputs(raw_x))
        train_y.extend(utils.preprocess_targets(raw_y))

        if client_idx <= len(client_ids_test) - 1:
            client_id_test = client_ids_test[client_idx]
            raw_tokens_test = test_h5[_EXAMPLE][client_id_test][_TOKENS][()]
            raw_tokens_test = [x.decode('utf8') for x in raw_tokens_test]
            raw_title_test = test_h5[_EXAMPLE][client_id_test][_TITLE][()]
            raw_title_test = [x.decode('utf8') for x in raw_title_test]
            raw_x_test = [
                ' '.join(pair) for pair in zip(raw_tokens_test, raw_title_test)
            ]
            raw_y_test = [
                x.decode('utf8') for x in test_h5[_EXAMPLE][client_id_test][_TAGS][()]
            ]
            test_x.extend(utils.preprocess_inputs(raw_x_test))
            test_y.extend(utils.preprocess_targets(raw_y_test))

    train_x, train_y = np.asarray(train_x).astype(np.float32), np.asarray(train_y).astype(np.float32)
    train_ds = data.TensorDataset(torch.tensor(train_x[:, :]),
                                  torch.tensor(train_y[:]))
    train_dl = data.DataLoader(dataset=train_ds,
                               batch_size=train_bs,
                               shuffle=True,
                               drop_last=False)

    if len(test_x) != 0:
        test_x, test_y = np.asarray(test_x).astype(np.float32), np.asarray(test_y).astype(np.float32)
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


def load_partition_data_distributed_federated_stackoverflow_lr(
        process_id, dataset, data_dir, batch_size=DEFAULT_BATCH_SIZE):

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
    output_dim = len(utils.get_tag_dict()) + 1 #oov
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, output_dim


def load_partition_data_federated_stackoverflow_lr(dataset, data_dir, batch_size=DEFAULT_BATCH_SIZE):
    logging.info("load_partition_data_federated_stackoverflow_lr START")
    
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

    output_dim = len(utils.get_tag_dict()) + 1 #oov
    return DEFAULT_TRAIN_CLINETS_NUM, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, output_dim


if __name__ == "__main__":
    # load_partition_data_federated_stackoverflow(None, None, 100, 128)
    train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, output_dim= load_partition_data_distributed_federated_stackoverflow_lr(
        2, None, None, 128)
    DEFAULT_TRAIN_CLINETS_NUM, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, output_dim = load_partition_data_federated_stackoverflow_lr(
        None, None, 128)
    print(train_data_local, test_data_local)
