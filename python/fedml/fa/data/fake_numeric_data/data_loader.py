import os
import random

from fedml.fa.data.utils import read_data, equally_partition_a_dataset


def generate_fake_data(data_cache_dir):
    file_path = data_cache_dir + "/fake_numeric_data.txt"

    if not os.path.exists(file_path):
        f = open(file_path, "a")
        for i in range(10000):
            f.write(f"{random.randint(1, 100)}\n")
        f.close()


def load_partition_data_fake(data_dir, client_num):
    dataset = read_data(data_dir=data_dir)
    return equally_partition_a_dataset(client_num, dataset)

