import math
import os


def equally_partition_a_dataset(client_num_in_total, dataset):
    """
    Equally partition a dataset among clients.

    Args:
        client_num_in_total (int): The total number of clients.
        dataset (list): The dataset to partition.

    Returns:
        tuple: A tuple containing the total dataset size, a dictionary of local data counts per client,
               and a dictionary of local data for each client.
    """
    client_data_num = int(len(dataset) / client_num_in_total)
    local_data_dict = dict()
    train_data_local_num_dict = dict()
    start_counter = 0
    datasize = len(dataset)
    for i in range(client_num_in_total):
        local_data_dict[i] = dataset[start_counter:start_counter + client_data_num]
        start_counter += client_data_num
        train_data_local_num_dict[i] = client_data_num
    return (
        datasize,
        train_data_local_num_dict,
        local_data_dict,
    )


def equally_partition_a_dataset_according_to_users(client_num_in_total, dataset):
    """
    Equally partition a dataset among clients based on the number of users.

    Args:
        client_num_in_total (int): The total number of clients.
        dataset (dict): The dataset organized by user IDs.

    Returns:
        tuple: A tuple containing the total dataset size, a dictionary of local data counts per client,
               and a dictionary of local data for each client.
    """
    user_num_for_one_client = int(math.ceil(len(dataset) / client_num_in_total))
    local_data_dict = dict()
    train_data_local_num_dict = dict()
    datasize = 0
    user_list = list(dataset.keys())
    user_list_counter = 0
    for i in range(client_num_in_total):
        local_data_dict[i] = list()
        client_data_num = 0
        for j in range(user_num_for_one_client):
            if user_list_counter >= len(user_list):
                break
            local_data_dict[i].extend(dataset[user_list[user_list_counter]])
            client_data_num += len(dataset[user_list[user_list_counter]])
            user_list_counter += 1
        train_data_local_num_dict[i] = client_data_num
        datasize += train_data_local_num_dict[i]
    return (
        datasize,
        train_data_local_num_dict,
        local_data_dict,
    )


def read_data(data_dir):
    """
    Read data from text files in a directory.

    Args:
        data_dir (str): The path to the directory containing text data files.

    Returns:
        list: A list of integers representing the dataset.
    """
    train_files = os.listdir(data_dir)
    train_files = [f for f in train_files if f.endswith(".txt")]
    dataset = []
    for f in train_files:
        file_path = os.path.join(data_dir, f)
        f2 = open(file_path, "r")
        lines = [int(line.strip()) for line in f2]
        dataset.extend(lines)
    return dataset


def read_data_with_column_idx(file_folder_path, column_idx, separator=","):
    """
    Read data from text files in a directory, selecting a specific column.

    Args:
        file_folder_path (str): The path to the directory containing text data files.
        column_idx (int): The index of the column to extract.
        separator (str, optional): The separator used in the text files (default is comma).

    Returns:
        list: A list of values from the selected column.
    """
    train_files = os.listdir(file_folder_path)
    train_files = [f for f in train_files if not f.startswith(".")]
    dataset = []
    for f in train_files:
        file_path = os.path.join(file_folder_path, f)
        f2 = open(file_path, "r")
        for line in f2:
            if len(line.split(separator)[column_idx].strip()) > 0:
                dataset.append(line.split(separator)[column_idx].strip())
    return dataset
