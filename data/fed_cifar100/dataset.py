import logging

import h5py
import numpy as np

import tensorflow_federated as tff
import tensorflow_datasets as tfds

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

download = True

def download_and_save_federated_cifar100(train_ds_path = './cifar100_train.h5', test_ds_path = './cifar100_test.h5'):
    
    cifar100_train, cifar100_test = tff.simulation.datasets.cifar100.load_data()
    
    cifar100_train_ds = [(k, data) for k in cifar100_train.client_ids for data in tfds.as_numpy(cifar100_train.create_tf_dataset_for_client(k))]
    logging.info("train dataset length : " + str(len(cifar100_train_ds)))
    
    train_h5 = h5py.File(train_ds_path, 'w')
    train_h5.create_dataset('id', data = np.string_([i[0] for i in cifar100_train_ds])) 
    train_h5.create_dataset('image', data = [i[1]['image'] for i in cifar100_train_ds])
    train_h5.create_dataset('label', data = [i[1]['label'] for i in cifar100_train_ds])
    train_h5.close()
    
    cifar100_test_ds = [(k, data) for k in cifar100_test.client_ids for data in tfds.as_numpy(cifar100_test.create_tf_dataset_for_client(k))]
    logging.info("test dataset length : " + str(len(cifar100_test_ds)))
    
    test_h5 = h5py.File(test_ds_path, 'w')
    test_h5.create_dataset('id', data = np.string_([i[0] for i in cifar100_test_ds])) 
    test_h5.create_dataset('image', data = [i[1]['image'] for i in cifar100_test_ds])
    test_h5.create_dataset('label', data = [i[1]['label'] for i in cifar100_test_ds])
    test_h5.close()
    
"""
#with Tensorflow dependencies, you can run this python script to process the data from Tensorflow Federated locally:
python dataset.py

Before downloading, please install TFF as its official instruction:
pip install --upgrade tensorflow_federated
"""
if __name__ == "__main__":
    if download:
        download_and_save_federated_cifar100()