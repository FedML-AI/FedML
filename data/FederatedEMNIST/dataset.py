import logging

import h5py
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_federated as tff

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

download = False
only_digit = False


def download_and_save_federated_emnist(train_ds_path='./emnist_train.h5', test_ds_path='./emnist_test.h5'):
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(only_digits=only_digit)

    emnist_train_ds = [(k, data) for k in emnist_train.client_ids for data in
                       tfds.as_numpy(emnist_train.create_tf_dataset_for_client(k))]
    logging.info("train dataset length : " + str(len(emnist_train_ds)))

    train_h5 = h5py.File(train_ds_path, 'w')
    train_h5.create_dataset('id', data=np.string_([i[0] for i in emnist_train_ds]))
    train_h5.create_dataset('pixels', data=[i[1]['pixels'] for i in emnist_train_ds])
    train_h5.create_dataset('label', data=[i[1]['label'] for i in emnist_train_ds])
    train_h5.close()

    emnist_test_ds = [(k, data) for k in emnist_test.client_ids for data in
                      tfds.as_numpy(emnist_test.create_tf_dataset_for_client(k))]
    logging.info("test dataset length : " + str(len(emnist_test_ds)))

    test_h5 = h5py.File(test_ds_path, 'w')
    test_h5.create_dataset('id', data=np.string_([i[0] for i in emnist_test_ds]))
    test_h5.create_dataset('pixels', data=[i[1]['pixels'] for i in emnist_test_ds])
    test_h5.create_dataset('label', data=[i[1]['label'] for i in emnist_test_ds])
    test_h5.close()


"""
Or with Tensorflow dependencies, you can run this to process the data from Tensorflow locally:

```
python dataset.py
```
"""
if __name__ == "__main__":
    if download:
        download_and_save_federated_emnist()
