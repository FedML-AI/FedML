import logging

import h5py
import torch
import random
import torch.utils.data as data
import utils

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

client_map_train = None
client_map_test = None
train_file_path = '../../../data/stackoverflow/datasets/stackoverflow_train.h5'
test_file_path = '../../../data/stackoverflow/datasets/stackoverflow_test.h5'
heldout_file_path = '../../../data/stackoverflow/datasets/stackoverflow_held_out.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_TOKENS = 'tokens'

def get_client_map(client_map, client_id = None, client_num = None):
    if client_map == None:
        random.shuffle(client_id)
        client_map = {k:[client_id[i] for i in range(k, len(client_id), client_num)] for k in range(client_num)}
    return client_map

def get_dataloader(dataset, data_dir, train_bs, test_bs, client_idx = None):
    train_h5 = h5py.File(train_file_path, 'r')
    test_h5 = h5py.File(test_file_path,'r')
    if client_idx is None:
        client_ids_train = list(train_h5[_EXAMPLE].keys())
        client_ids_test = list(test_h5[_EXAMPLE].keys())
    else:
        client_ids_train = get_client_map(client_map_train)[client_idx]
        client_ids_test = get_client_map(client_map_test)[client_idx]
        
    train_ds = []
    test_ds = []
    for client_id in client_ids_train:
        raw_train = train_h5[_EXAMPLE][client_id][_TOKENS][()]
        raw_train = [x.decode('utf8') for x in raw_train]
        train_ds.extend(utils.preprocess(raw_train))
    for client_id in client_ids_test:
        raw_test = test_h5[_EXAMPLE][client_id][_TOKENS][()]
        raw_test = [x.decode('utf8') for x in raw_test]
        test_ds.extend(utils.preprocess(raw_test))
    train_x, train_y = utils.split(train_ds)
    test_x, test_y = utils.split(test_ds)
    train_ds = data.TensorDataset(torch.tensor(train_x[:,:]), torch.tensor(train_y[:]))
    test_ds = data.TensorDataset(torch.tensor(test_x[:,:]), torch.tensor(test_y[:]))
    train_dl = data.DataLoader(dataset = train_ds, batch_size=train_bs, shuffle = True, drop_last = False)
    test_dl = data.DataLoader(dataset = test_ds, batch_size=test_bs, shuffle = True, drop_last = False)
    
    train_h5.close()
    test_h5.close()
    return train_dl, test_dl

def load_partition_data_distributed_federated_stackoverflow(process_id, dataset, data_dir, client_number, batch_size):
    
    # get global dataset
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size, process_id - 1)
        train_data_num = len(train_data_global)
        test_data_num = len(test_data_global)
        logging.info("train_dl_global number = " + str(train_data_num))
        logging.info("test_dl_global number = " + str(test_data_num))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        train_h5 = h5py.File(train_file_path, 'r')
        test_h5 = h5py.File(test_file_path, 'r')
        global client_map_train, client_map_test
        client_map_train = get_client_map(client_map_train, list(train_h5[_EXAMPLE].keys()), client_number)
        client_map_test = get_client_map(client_map_test, list(test_h5[_EXAMPLE].keys()), client_number)
        train_h5.close()
        test_h5.close()
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, process_id - 1)
        train_data_num = local_data_num = len(train_data_local) + len(test_data_local)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local



def load_partition_data_federated_stackoverflow(dataset, data_dir, client_number, batch_size):
    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    train_data_num = len(train_data_global)
    test_data_num = len(test_data_global)
        
    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_h5 = h5py.File(train_file_path, 'r')
    test_h5 = h5py.File(test_file_path, 'r')
    global client_map_train, client_map_test
    client_map_train = get_client_map(client_map_train, list(train_h5[_EXAMPLE].keys()), client_number)
    client_map_test = get_client_map(client_map_test, list(test_h5[_EXAMPLE].keys()), client_number)
    train_h5.close()
    test_h5.close()
    
    for client_idx in range(client_number):
    
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, client_idx)
        local_data_num = len(train_data_local) + len(test_data_local)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    
    return train_data_num, test_data_num, train_data_global, test_data_global, \
        data_local_num_dict, train_data_local_dict, test_data_local_dict
        
if __name__ == "__main__":
    #load_partition_data_federated_stackoverflow(None, None, 100, 128)
    train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local = load_partition_data_distributed_federated_stackoverflow(2, None, None, 1000, 128)
    print(train_data_local, test_data_local)