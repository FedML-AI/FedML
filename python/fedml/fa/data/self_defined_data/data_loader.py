import logging
import os
import random

from fedml.fa.data.utils import read_data, equally_partition_a_dataset, read_data_with_column_idx


def generate_fake_data(data_cache_dir):
    file_path = data_cache_dir + "/fake_numeric_data.txt"

    if not os.path.exists(file_path):
        f = open(file_path, "a")
        for i in range(10000):
            f.write(f"{random.randint(1, 100)}\n")
        f.close()


def load_partition_self_defined_data(file_folder_path, client_num, data_col_idx, separator=","):
    if not os.path.exists(file_folder_path):
        raise Exception(f"No data file: {file_folder_path}")
    logging.info(f"file_folder_path = {file_folder_path}")
    dataset = read_data_with_column_idx(file_folder_path=file_folder_path, column_idx=data_col_idx, seperator=separator)
    return equally_partition_a_dataset(client_num, dataset)

