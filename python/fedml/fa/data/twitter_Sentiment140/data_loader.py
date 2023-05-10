import os
import pickle
import zipfile
import wget
from fedml.fa.constants import FA_DATA_TWITTER_Sentiment140_URL
from fedml.fa.data.utils import equally_partition_a_dataset, equally_partition_a_dataset_according_to_users


def download_twitter_Sentiment140(data_cache_dir):
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)
    file_path = os.path.join(data_cache_dir, "trainingandtestdata.zip")

    if not os.path.exists(file_path): # Download the file (if we haven't already)
        wget.download(FA_DATA_TWITTER_Sentiment140_URL, out=file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_cache_dir)


def load_partition_data_twitter_sentiment140(dataset, client_num_in_total):
    return equally_partition_a_dataset_according_to_users(client_num_in_total, dataset)

