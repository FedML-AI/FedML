import csv

import numpy as np
from sklearn.cluster import KMeans


class DataLoader(object):
    """
    DataLoader class for managing data loading and preprocessing.

    Args:
        data_name (str): The name of the dataset.
        data_path (str): The path to the dataset CSV file.
        client_list (list): A list of client IDs.
        sample_num_in_total (int): The total number of data samples.
        beta (float): A parameter for data loading.

    Attributes:
        data_name (str): The name of the dataset.
        data_path (str): The path to the dataset CSV file.
        client_list (list): A list of client IDs.
        sample_num_in_total (int): The total number of data samples.
        beta (float): A parameter for data loading.
        streaming_full_dataset_X (list): A list to store data samples.
        streaming_full_dataset_Y (list): A list to store data labels.
        StreamingDataDict (dict): A dictionary to store streaming data for clients.

    Methods:
        load_datastream():
            Load and preprocess the data for streaming and return it as a dictionary.
        load_adversarial_data():
            Load adversarial data based on the beta parameter.
        load_stochastic_data():
            Load stochastic data based on the beta parameter.
        read_csv_file(percent):
            Read and return data samples and labels from a CSV file.
        read_csv_file_for_cluster(percent):
            Read and cluster data samples based on the beta parameter.
        kMeans(X):
            Perform K-means clustering on the data.
        preprocessing():
            Perform preprocessing on the data.

    """
    def __init__(self, data_name, data_path, client_list, sample_num_in_total, beta):
        """
        Initialize the DataLoader with dataset information and parameters.

        Args:
            data_name (str): The name of the dataset.
            data_path (str): The path to the dataset CSV file.
            client_list (list): A list of client IDs.
            sample_num_in_total (int): The total number of data samples.
            beta (float): A parameter for data loading.

        Note:
            This constructor initializes the DataLoader with dataset details and parameters.

        Returns:
            None
        """
        # SUSY, Room Occupancy;
        self.data_name = data_name
        self.data_path = data_path
        self.client_list = client_list
        self.sample_num_in_total = sample_num_in_total
        self.beta = beta
        self.streaming_full_dataset_X = []
        self.streaming_full_dataset_Y = []
        self.StreamingDataDict = {}

    """
        return streaming_data
            key: client_id
            value: [sample1, sample2, ..., sampleN]
                    sample: {"x": [1,2,3,4,5,...,M]; "y":0}
    """

    def load_datastream(self):
        """
        Load and preprocess the data for streaming and return it as a dictionary.

        Returns:
            dict: A dictionary containing streaming data for clients.
        """
        self.preprocessing()
        self.load_adversarial_data()
        self.load_stochastic_data()
        # for value in self.StreamingDataDict.values():
        #     random.shuffle(value)

        # for client_index in self.client_list:
        #     length = len(self.StreamingDataDict[client_index])
        #     logging.info("len of index %d = %d" % (client_index, length))
        return self.StreamingDataDict

    # beta (clustering, GMM)
    def load_adversarial_data(self):
        """
        Load adversarial data based on the beta parameter.

        Returns:
            dict: A dictionary containing adversarial streaming data for clients.
        """
        streaming_data = self.read_csv_file_for_cluster(self.beta)
        return streaming_data

    def load_stochastic_data(self):
        """
        Load stochastic data based on the beta parameter.

        Returns:
            dict: A dictionary containing stochastic streaming data for clients.
        """
        streaming_data = self.read_csv_file(self.beta)
        return streaming_data

    def read_csv_file(self, percent):
        """
        Read and return data samples and labels from a CSV file.

        Args:
            percent (float): The percentage of data to read.

        Returns:
            dict: A dictionary containing streaming data for clients.
        """

        # print("start from:")
        iteration_number = int(self.sample_num_in_total / len(self.client_list))
        index_start = int(percent * self.sample_num_in_total)
        stochastic_data_x = []
        stochastic_data_y = []
        for i_x, dp_x in enumerate(self.streaming_full_dataset_X):
            if i_x >= index_start:
                stochastic_data_x.append(dp_x)
        for i_y, dp_y in enumerate(self.streaming_full_dataset_Y):
            if i_y >= index_start:
                stochastic_data_y.append(dp_y)
        for c_index in self.client_list:
            if (
                len(self.StreamingDataDict[self.client_list[c_index]])
                > iteration_number
            ):
                for i, data_point in enumerate(
                    self.StreamingDataDict[self.client_list[c_index]]
                ):
                    if i >= iteration_number:
                        stochastic_data_x.append(data_point["x"])
                        stochastic_data_y.append(data_point["y"])
                self.StreamingDataDict[
                    self.client_list[c_index]
                ] = self.StreamingDataDict[self.client_list[c_index]][
                    0:iteration_number
                ]

        # print("***")
        # for c_index in self.client_list:
        #     print(len(self.StreamingDataDict[self.client_list[c_index]]))
        # print("***")
        client_index = 0
        full_count = 0
        # print("iteration_number = " + str(iteration_number))
        # print("len stochastic_data_x = " + str(len(stochastic_data_x)))
        for i in range(len(stochastic_data_x)):
            while (
                len(self.StreamingDataDict[self.client_list[client_index]])
                == iteration_number
            ):
                client_index += 1
                full_count += 1
            sample = {}
            sample["x"] = stochastic_data_x[i]
            sample["y"] = stochastic_data_y[i]
            self.StreamingDataDict[self.client_list[client_index]].append(sample)
            if (
                len(self.StreamingDataDict[self.client_list[client_index]])
                == iteration_number
                and full_count == len(self.client_list) - 1
            ):
                full_count += 1
            if full_count == len(self.client_list):
                # print("stop at index = " + str(i))
                break
        return self.StreamingDataDict

    def read_csv_file_for_cluster(self, percent):
        """
        Read and cluster data samples based on the beta parameter.

        Args:
            percent (float): The percentage of data to read and cluster.

        Returns:
            dict: A dictionary containing clustered streaming data for clients.
        """
        data = []
        label = []
        for client_id in self.client_list:
            self.StreamingDataDict[client_id] = []
        if percent == 0:
            return self.StreamingDataDict

        for i, row in enumerate(self.streaming_full_dataset_X):
            if i >= (self.sample_num_in_total * percent):
                break
            data.append(self.streaming_full_dataset_X[i])
            label.append(self.streaming_full_dataset_Y[i])
        # print("Clustering Started")
        clusters = self.kMeans(data)
        # print("Clustering Finished")

        for i, cluster in enumerate(clusters):
            sample = {}
            sample["y"] = label[i]
            sample["x"] = data[i]
            self.StreamingDataDict[self.client_list[cluster]].append(sample)
        # print("Arrange Clustered Data has Finished")

        # for id in self.client_list:
        # print("after clustering:")
        # print(len(self.StreamingDataDict[self.client_list[id]]))
        return self.StreamingDataDict

    def kMeans(self, X):
        """
        Perform K-means clustering on the data.

        Args:
            X (list): List of data samples.

        Returns:
            array: Cluster labels for data samples.
        """
        kmeans = KMeans(n_clusters=len(self.client_list))
        kmeans.fit(X)
        return kmeans.labels_

    def preprocessing(self):
        """
        Perform preprocessing on the data.

        Returns:
            None
        """
        # print("sample_num_in_total = " + str(self.sample_num_in_total))
        data = []
        with open(self.data_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(readCSV):
                if i < self.sample_num_in_total:
                    if self.data_name == "SUSY":
                        data.append(np.asarray(row[1:], dtype=np.float32))
                        self.streaming_full_dataset_Y.append(int(row[0].split(".")[0]))
                    elif self.data_name == "RO":
                        data.append(np.asarray(row[2:-1], dtype=np.float32))
                        self.streaming_full_dataset_Y.append(int(row[-1].split(".")[0]))
        # min_max_scaler = preprocessing.MinMaxScaler()
        # self.streaming_full_dataset_X = min_max_scaler.fit_transform(data)
        self.streaming_full_dataset_X = data
        # print("############")
        # print(len(self.streaming_full_dataset_X))
        # print(self.streaming_full_dataset_X)
