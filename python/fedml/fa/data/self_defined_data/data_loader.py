import logging
import os
import random

from fedml.fa.data.utils import read_data, equally_partition_a_dataset, read_data_with_column_idx


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
    file_path = data_cache_dir + "/fake_numeric_data.txt"

    if not os.path.exists(file_path):
        f = open(file_path, "a")
        for i in range(10000):
            f.write(f"{random.randint(1, 100)}\n")
        f.close()


def load_partition_self_defined_data(file_folder_path, client_num, data_col_idx, separator=","):
    """
    Load and partition self-defined data from a text file into client-specific partitions.

    Args:
        file_folder_path (str): The path to the text file containing the data.
        client_num (int): The total number of clients to partition the data for.
        data_col_idx (int): The column index of the data to be used.
        separator (str): The separator used in the data file (default is comma ',').

    Raises:
        Exception: If the specified data file does not exist.

    Returns:
        tuple: A tuple containing the dataset size, a dictionary of client data sizes, and a dictionary of client data.
    """
    if not os.path.exists(file_folder_path):
        raise Exception(f"No data file: {file_folder_path}")
    logging.info(f"file_folder_path = {file_folder_path}")
    dataset = read_data_with_column_idx(file_folder_path=file_folder_path, column_idx=data_col_idx, separator=separator)
    return equally_partition_a_dataset(client_num, dataset)
