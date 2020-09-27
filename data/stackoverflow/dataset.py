import logging

import h5py
import random
import numpy as np

import tensorflow_federated as tff
import tensorflow_datasets as tfds

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def download_and_save_stackoverflow(train_ds_path = './stackoverflow_train.h5', test_ds_path = './stackoverflow_train.h5'):
    
    #synthetic = tff.simulation.datasets.stackoverflow.get_synthetic()
    #print(synthetic)
    # s = tff.simulation.datasets.stackoverflow.load_data(cache_dir='d:/')
    # stackoverflow_train, stackoverflow_heldout, stackoverflow_test = tff.simulation.datasets.stackoverflow.load_data(cache_dir='d:/')
    
    # logging.info("loaded data from tff")
    
    # stackoverflow_train_ds = [(k, data) for k in stackoverflow_train.client_ids for data in tfds.as_numpy(stackoverflow_train.create_tf_dataset_for_client(k))]
    # logging.info("train dataset length : " + str(len(stackoverflow_train_ds)))
    
    # train_h5 = h5py.File(train_ds_path, 'w')
    # train_h5.create_dataset('id', data = np.string_([i[0] for i in stackoverflow_train_ds])) 
    # train_h5.create_dataset('creation_date', data = [i[1]['creation_date'] for i in stackoverflow_train_ds])
    # train_h5.create_dataset('title', data = [i[1]['title'] for i in stackoverflow_train_ds])
    # train_h5.create_dataset('score', data = [i[1]['score'] for i in stackoverflow_train_ds])
    # train_h5.create_dataset('tags', data = [i[1]['tags'] for i in stackoverflow_train_ds])
    # train_h5.create_dataset('tokens', data = [i[1]['tokens'] for i in stackoverflow_train_ds])
    # train_h5.create_dataset('type', data = [i[1]['type'] for i in stackoverflow_train_ds])
    # train_h5.close()
    
    # stackoverflow_test_ds = [(k, data) for k in stackoverflow_test.client_ids for data in tfds.as_numpy(stackoverflow_test.create_tf_dataset_for_client(k))]
    # logging.info("test dataset length : " + str(len(stackoverflow_test_ds)))
    
    # test_h5 = h5py.File(test_ds_path, 'w')
    # test_h5.create_dataset('id', data = np.string_([i[0] for i in stackoverflow_test_ds])) 
    # test_h5.create_dataset('creation_date', data = [i[1]['creation_date'] for i in stackoverflow_test_ds])
    # test_h5.create_dataset('title', data = [i[1]['title'] for i in stackoverflow_test_ds])
    # test_h5.create_dataset('score', data = [i[1]['score'] for i in stackoverflow_test_ds])
    # test_h5.create_dataset('tags', data = [i[1]['tags'] for i in stackoverflow_test_ds])
    # test_h5.create_dataset('tokens', data = [i[1]['tokens'] for i in stackoverflow_test_ds])
    # test_h5.create_dataset('type', data = [i[1]['type'] for i in stackoverflow_test_ds])
    # test_h5.close()
    '''
    from tensorflow_federated.python.simulation import hdf5_client_data
    from tqdm import tqdm
    stackoverflow_heldout_ds = list()
    stackoverflow_heldout = hdf5_client_data.HDF5ClientData('C:/Users/ellie/.keras/datasets/stackoverflow_held_out.h5')
    for k in tqdm(stackoverflow_heldout.client_ids):
        for data in tfds.as_numpy(stackoverflow_heldout.create_tf_dataset_for_client(k)):
                stackoverflow_heldout_ds.append((k, data))
    #stackoverflow_heldout_ds = [(k, data) for k in stackoverflow_heldout.client_ids for data in tfds.as_numpy(stackoverflow_heldout.create_tf_dataset_for_client(k))]
    logging.info("heldout dataset length : " + str(len(stackoverflow_heldout_ds)))
    
    heldout_h5 = h5py.File(stackoverflow_heldout_ds, 'w')
    heldout_h5.create_dataset('id', data = np.string_([i[0] for i in stackoverflow_heldout_ds])) 
    heldout_h5.create_dataset('creation_date', data = [i[1]['creation_date'] for i in stackoverflow_heldout_ds])
    heldout_h5.create_dataset('title', data = [i[1]['title'] for i in stackoverflow_heldout_ds])
    heldout_h5.create_dataset('score', data = [i[1]['score'] for i in stackoverflow_heldout_ds])
    heldout_h5.create_dataset('tags', data = [i[1]['tags'] for i in stackoverflow_heldout_ds])
    heldout_h5.create_dataset('tokens', data = [i[1]['tokens'] for i in stackoverflow_heldout_ds])
    heldout_h5.create_dataset('type', data = [i[1]['type'] for i in stackoverflow_heldout_ds])
    heldout_h5.close()
    '''
    
if __name__ == "__main__":
    download_and_save_stackoverflow()