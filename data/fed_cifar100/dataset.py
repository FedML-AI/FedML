import logging

import tensorflow_federated as tff

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

download = True

def download_and_save_federated_cifar100():
    cifar100_train, cifar100_test = tff.simulation.datasets.cifar100.load_data(cache_dir='./')
    
    
"""
#with Tensorflow dependencies, you can run this python script to process the data from Tensorflow Federated locally:
python dataset.py

Before downloading, please install TFF as its official instruction:
pip install --upgrade tensorflow_federated
"""
if __name__ == "__main__":
    if download:
        download_and_save_federated_cifar100()