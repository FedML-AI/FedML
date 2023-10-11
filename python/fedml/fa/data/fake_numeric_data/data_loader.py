import os
import random

from fedml.fa.data.utils import read_data, equally_partition_a_dataset


def generate_fake_data(data_cache_dir):
    """
    Generate fake numeric data and save it to a text file in the specified directory.

    Args:
        data_cache_dir (str): The directory where the fake numeric data file should be saved.

    Note:
        This function generates random integer data and writes it to a text file.

    Returns:
        None
    """
    file_path = os.path.join(data_cache_dir, "fake_numeric_data.txt")

    if not os.path.exists(file_path):
        with open(file_path, "a") as f:
            for i in range(10000):
                f.write(f"{random.randint(1, 100)}\n")


def load_partition_data_fake(data_dir, client_num):
    """
    Load and partition fake data from a specified directory into client-specific partitions.

    Args:
        data_dir (str): The directory path where the fake data is located.
        client_num (int): The total number of clients to partition the data for.

    Returns:
        tuple: A tuple containing the dataset size, a dictionary of client data sizes, and a dictionary of client data.
    """
    dataset = read_data(data_dir=data_dir)
    return equally_partition_a_dataset(client_num, dataset)
