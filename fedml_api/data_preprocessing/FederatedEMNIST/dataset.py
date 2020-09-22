import logging

import h5py
import random
import numpy as np

import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_datasets as tfds

from data_loader import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

download = False

def download_and_save_federated_emnist(train_ds_path = './emnist_train.h5', test_ds_path='./emnist_test.h5'):
    
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    
    emnist_train_ds = [(k, data) for k in emnist_train.client_ids for data in tfds.as_numpy(emnist_train.create_tf_dataset_for_client(k))]
    logging.info("train dataset length : " + str(len(emnist_train_ds)))
    
    train_h5 = h5py.File(train_ds_path, 'w')
    train_h5.create_dataset('id', data = np.string_([i[0] for i in emnist_train_ds])) 
    train_h5.create_dataset('pixels', data = [i[1]['pixels'] for i in emnist_train_ds])
    train_h5.create_dataset('label', data = [i[1]['label'] for i in emnist_train_ds])
    train_h5.close()
    
    emnist_test_ds = [(k, data) for k in emnist_test.client_ids for data in tfds.as_numpy(emnist_test.create_tf_dataset_for_client(k))]
    logging.info("test dataset length : " + str(len(emnist_test_ds)))
    
    test_h5 = h5py.File(test_ds_path, 'w')
    test_h5.create_dataset('id', data = np.string_([i[0] for i in emnist_test_ds])) 
    test_h5.create_dataset('pixels', data = [i[1]['pixels'] for i in emnist_test_ds])
    test_h5.create_dataset('label', data = [i[1]['label'] for i in emnist_test_ds])
    test_h5.close()
    

def test_federated_emnist():
    
    client_num = 300
    test_num = 10 # use 'test_num = client_num' to test on all generated client dataset
    
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()    
    client_train_ds = list(iter(tfds.as_numpy(emnist_train.create_tf_dataset_from_all_clients())))
    client_test_ds = list(iter(tfds.as_numpy(emnist_test.create_tf_dataset_from_all_clients())))
    
    _, _, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict, _ = load_partition_data_federated_emnist(None, None, client_num, 1)
    client_train_dl = list(iter(train_data_global))
    client_test_dl = list(iter(test_data_global))
    
    assert(len(client_train_ds) == len(client_train_dl))
    assert(len(client_test_ds) == len(client_test_dl))
    
    for idx in random.sample(range(client_num),test_num):
        train_local_dl = list(iter(train_data_local_dict[idx]))
        train_local_dl = {str(dl[0].numpy().squeeze()): dl[1].numpy().squeeze() for dl in train_local_dl}
        test_local_dl = list(iter(test_data_local_dict[idx]))
        test_local_dl = {str(dl[0].numpy().squeeze()): dl[1].numpy().squeeze() for dl in test_local_dl}
        for client_id in get_clent_map()[idx]:
            train_local_ds = list(iter(tfds.as_numpy(emnist_train.create_tf_dataset_for_client(client_id.decode("utf-8")))))
            train_local_ds = [(str(ds['pixels']), ds['label']) for ds in train_local_ds]
            for ds in train_local_ds:
                assert(ds[0] in train_local_dl)
                assert(ds[1] == train_local_dl[ds[0]])
                
            test_local_ds = list(iter(tfds.as_numpy(emnist_test.create_tf_dataset_for_client(client_id.decode("utf-8")))))
            test_local_ds = [(str(ds['pixels']), ds['label']) for ds in test_local_ds]
            for ds in test_local_ds:
                assert(ds[0] in test_local_dl)
                assert(ds[1] == test_local_dl[ds[0]])
                
        logging.info("Test for dataset on client = %d passed."%idx)

    logging.info("Tests for dataset passed.")

if __name__ == "__main__":
    if download:
        download_and_save_federated_emnist()
    test_federated_emnist()