import logging
import math
import os
import pickle
import zipfile
import wget
from fedml.fa.constants import FA_DATA_TWITTER_Sentiment140_URL
from fedml.fa.data.utils import equally_partition_a_dataset, equally_partition_a_dataset_according_to_users


def download_twitter_Sentiment140(data_cache_dir):
    """
    Download the Sentiment140 Twitter dataset if it doesn't exist in the specified directory.

    Args:
        data_cache_dir (str): The directory where the dataset should be downloaded.

    Note:
        This function downloads the dataset from a URL and extracts it to the specified directory.

    Returns:
        None
    """
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir, exist_ok=True)
    file_path = os.path.join(data_cache_dir, "trainingandtestdata.zip")

    if not os.path.exists(file_path):  # Download the file (if we haven't already)
        wget.download(FA_DATA_TWITTER_Sentiment140_URL, out=file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_cache_dir)


def load_partition_data_twitter_sentiment140(dataset, client_num_in_total):
    """
    Load and partition the Sentiment140 Twitter dataset into client-specific partitions.

    Args:
        dataset (dict): A dictionary containing client usernames as keys and their data as values.
        client_num_in_total (int): The total number of clients to partition the data for.

    Returns:
        tuple: A tuple containing the dataset size, a dictionary of client data sizes, and a dictionary of client data.
    """
    return equally_partition_a_dataset_according_to_users(client_num_in_total, dataset)


def load_partition_data_twitter_sentiment140_heavy_hitter(dataset, client_num_in_total):
    """
    Load and partition the Sentiment140 Twitter dataset for heavy hitters into client-specific partitions.

    Args:
        dataset (dict): A dictionary containing client usernames as keys and their data as values.
        client_num_in_total (int): The total number of clients to partition the data for.

    Returns:
        tuple: A tuple containing the dataset size, a dictionary of client data sizes, and a dictionary of client data.
    """
    local_data_dict = dict()
    train_data_local_num_dict = dict()
    heavy_hitters = list(dataset.values())
    datasize = len(heavy_hitters)
    n = int(math.ceil(len(heavy_hitters) * 1.0 / client_num_in_total))  # data_size_for_each_client

    for user_idx in range(0, client_num_in_total - 1):
        local_data_dict[user_idx] = heavy_hitters[user_idx * n: (user_idx + 1) * n]
        train_data_local_num_dict[user_idx] = n
    local_data_dict[client_num_in_total - 1] = heavy_hitters[(client_num_in_total - 1) * n: datasize]
    train_data_local_num_dict[client_num_in_total - 1] = datasize - (client_num_in_total - 1) * n

    return (
        datasize,
        train_data_local_num_dict,
        local_data_dict,
    )
