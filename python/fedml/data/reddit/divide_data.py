import csv
import logging
import random
import time
from collections import defaultdict
from random import Random

import numpy as np
from torch.utils.data import DataLoader

#from argParser import args


class Partition(object):
    """
    Helper class for dataset partitioning.

    Args:
        data (list): The dataset to be partitioned.
        index (list): A list of indices specifying the partition.

    Attributes:
        data (list): The dataset to be partitioned.
        index (list): A list of indices specifying the partition.

    Methods:
        __len__():
            Get the length of the partition.
        __getitem__(index):
            Get an item from the partition by index.

    """

    def __init__(self, data, index):
        """
        Initialize a dataset partition.

        Args:
            data (list): The dataset to be partitioned.
            index (list): A list of indices specifying the partition.

        Returns:
            None
        """
        self.data = data
        self.index = index

    def __len__(self):
        """
        Get the length of the partition.

        Returns:
            int: The length of the partition.
        """
        return len(self.index)

    def __getitem__(self, index):
        """
        Get an item from the partition by index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            object: The item from the partition.
        """
        data_idx = self.index[index]
        return self.data[data_idx]


import csv
import logging
import numpy as np
from collections import defaultdict
from random import Random

class DataPartitioner(object):
    """
    Partition data by trace or random for federated learning.

    Args:
        data: The dataset to be partitioned.
        args: An object containing configuration parameters.
        numOfClass (int): The number of classes in the dataset (default: 0).
        seed (int): The seed for randomization (default: 10).
        isTest (bool): Whether the partitioning is for a test dataset (default: False).

    Attributes:
        partitions (list): A list of partitions, where each partition is a list of sample indices.
        rng (Random): A random number generator.
        data: The dataset to be partitioned.
        labels: The labels of the dataset.
        args: An object containing configuration parameters.
        isTest (bool): Whether the partitioning is for a test dataset.
        data_len (int): The length of the dataset.
        task: The task type.
        numOfLabels (int): The number of labels in the dataset.
        client_label_cnt (defaultdict): A dictionary to count labels for each client.

    Methods:
        getNumOfLabels():
            Get the number of unique labels in the dataset.
        getDataLen():
            Get the length of the dataset.
        getClientLen():
            Get the number of clients/partitions.
        getClientLabel():
            Get the number of unique labels for each client.
        trace_partition(data_map_file):
            Partition data based on a trace file.
        partition_data_helper(num_clients, data_map_file=None):
            Helper function for partitioning data.
        uniform_partition(num_clients):
            Uniformly partition data randomly.
        use(partition, istest):
            Get a partition of the dataset for a specific client.
        getSize():
            Get the size of each partition (number of samples).

    """
    def __init__(self, data, args, numOfClass=0, seed=10, isTest=False):
        """
        Initialize the DataPartitioner.

        Args:
            data: The dataset to be partitioned.
            args: An object containing configuration parameters.
            numOfClass (int): The number of classes in the dataset (default: 0).
            seed (int): The seed for randomization (default: 10).
            isTest (bool): Whether the partitioning is for a test dataset (default: False).

        Note:
            This constructor sets up the DataPartitioner with the provided dataset and configuration.

        Returns:
            None
        """
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)

        self.data = data
        self.labels = self.data.targets
        self.args = args
        self.isTest = isTest
        np.random.seed(seed)

        self.data_len = len(self.data)
        self.task = args.task
        self.numOfLabels = numOfClass
        self.client_label_cnt = defaultdict(set)

    def getNumOfLabels(self):
        """
        Get the number of unique labels in the dataset.

        Returns:
            int: The number of unique labels.
        """
        return self.numOfLabels

    def getDataLen(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.data_len

    def getClientLen(self):
        """
        Get the number of clients/partitions.

        Returns:
            int: The number of clients/partitions.
        """
        return len(self.partitions)

    def getClientLabel(self):
        """
        Get the number of unique labels for each client.

        Returns:
            list: A list of the number of unique labels for each client.
        """
        return [len(self.client_label_cnt[i]) for i in range(self.getClientLen())]

    def trace_partition(self, data_map_file):
        """
        Partition data based on a trace file.

        Args:
            data_map_file (str): The path to the data mapping file.

        Returns:
            None
        """
        logging.info(f"Partitioning data by profile {data_map_file}...")
        
        clientId_maps = {}
        unique_clientIds = {}

        # Load meta data from the data_map_file
        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0

            for row in csv_reader:
                if read_first:
                    logging.info(f'Trace names are {", ".join(row)}')
                    read_first = False
                else:
                    client_id = row[0]

                    if client_id not in unique_clientIds:
                        unique_clientIds[client_id] = len(unique_clientIds)

                    clientId_maps[sample_id] = unique_clientIds[client_id]
                    self.client_label_cnt[unique_clientIds[client_id]].add(row[-1])
                    sample_id += 1

        # Partition data given mapping
        self.partitions = [[] for _ in range(len(unique_clientIds))]

        for idx in range(sample_id):
            self.partitions[clientId_maps[idx]].append(idx)

    def partition_data_helper(self, num_clients, data_map_file=None):
        """
        Helper function for partitioning data.

        Args:
            num_clients (int): The number of clients/partitions.
            data_map_file (str): The path to the data mapping file (default: None).

        Returns:
            None
        """
        # Read mapping file to partition trace
        if data_map_file is not None:
            self.trace_partition(data_map_file)
        else:
            self.uniform_partition(num_clients=num_clients)

    def uniform_partition(self, num_clients):
        """
        Uniformly partition data randomly.

        Args:
            num_clients (int): The number of clients/partitions.

        Returns:
            None
        """
        # Random partition
        numOfLabels = self.getNumOfLabels()
        data_len = self.getDataLen()
        logging.info(f"Randomly partitioning data, {data_len} samples...")

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        for _ in range(num_clients):
            part_len = int(1. / num_clients * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition, istest):
        """
        Get a partition of the dataset for a specific client.

        Args:
            partition (int): The index of the client/partition.
            istest (bool): Whether the partition is for a test dataset.

        Returns:
            Partition: A partition of the dataset for the specified client.
        """
        resultIndex = self.partitions[partition]

        exeuteLength = len(resultIndex) if not istest else int(
            len(resultIndex) * self.args.test_ratio)
        resultIndex = resultIndex[:exeuteLength]
        self.rng.shuffle(resultIndex)

        return Partition(self.data, resultIndex)

    def getSize(self):
        """
        Get the size of each partition (number of samples).

        Returns:
            dict: A dictionary containing the size of each partition.
        """
        # Return the size of samples
        return {'size': [len(partition) for partition in self.partitions]}


def select_dataset(rank, partition, batch_size, args, isTest=False, collate_fn=None):
    """
    Load data for a specific client based on client ID.

    Args:
        rank (int): The client's rank or ID.
        partition (Partition): A partition of the dataset for the client.
        batch_size (int): The batch size for data loading.
        args: An object containing configuration parameters.
        isTest (bool): Whether the data loading is for a test dataset (default: False).
        collate_fn (callable, optional): A function used to collate data samples into batches (default: None).

    Returns:
        DataLoader: A DataLoader object for loading the client's data.
    """
    partition = partition.use(rank - 1, isTest)
    dropLast = False if isTest else True
    num_loaders = min(int(len(partition)/args.batch_size/2), args.num_loaders)
    if num_loaders == 0:
        time_out = 0
    else:
        time_out = 60

    # if collate_fn is not None:
    #     return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn)
    # return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast)

    if collate_fn is not None:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn)
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_loaders, drop_last=dropLast)

