import os
import pickle
import zipfile
import wget
from fedml.fa.constants import FA_DATA_TWITTER_Sentiment140_URL
from fedml.fa.data.utils import equally_partition_a_dataset


def download_twitter_Sentiment140(data_cache_dir):
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)
    file_path = os.path.join(data_cache_dir, "trainingandtestdata.zip")

    if not os.path.exists(file_path): # Download the file (if we haven't already)
        wget.download(FA_DATA_TWITTER_Sentiment140_URL, out=file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_cache_dir)


def load_partition_data_twitter_sentiment140(data_dir, client_num_in_total):
    clients_triehh_file = os.path.join(data_dir, 'clients_triehh.txt')
    with open(clients_triehh_file, 'rb') as fp:
        dataset = pickle.load(fp)
    return equally_partition_a_dataset(client_num_in_total, dataset)

