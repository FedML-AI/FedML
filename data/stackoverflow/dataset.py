import logging

import h5py
import random
import numpy as np

import tensorflow_federated as tff
import tensorflow_datasets as tfds

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def download_and_save_stackoverflow():
    tff.simulation.datasets.stackoverflow.load_data(cache_dir='./')
    
def download_and_save_wordcount():
    tff.simulation.datasets.stackoverflow.load_word_counts(cache_dir='./')
    
if __name__ == "__main__":
    # download_and_save_stackoverflow()
    download_and_save_wordcount()