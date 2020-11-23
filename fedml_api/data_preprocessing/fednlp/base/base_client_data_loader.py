from abc import ABC, abstractmethod
from fedml_api.data_preprocessing.fednlp.base.utils import SpacyTokenizer
import pickle


class BaseClientDataLoader(ABC):
    @abstractmethod
    def __init__(self, data_path, partition_path, client_idx, partition_method, tokenize, data_fields):
        self.data_path = data_path
        self.partition_path = partition_path
        self.client_idx = client_idx
        self.partition_method = partition_method
        self.tokenize = tokenize
        self.data_fields = data_fields
        self.train_data = None
        self.test_data = None
        self.load_data()
        if self.tokenize:
            self.spacy_tokenizer = SpacyTokenizer()

    def get_train_batch_data(self, batch_size=None):
        if batch_size is None:
            return self.train_data
        else:
            batch_data_list = list()
            start = 0
            while start < len(self.train_data):
                end = start + batch_size if start + batch_size < len(self.train_data) else len(self.train_data)
                batch_data = dict()
                for field in self.data_fields:
                    batch_data[field] = self.train_data[field][start: end]
                    batch_data_list.append(batch_data)
                start = end
            return batch_data_list

    def get_test_batch_data(self, batch_size=None):
        if batch_size is None:
            return self.test_data
        else:
            batch_data_list = list()
            start = 0
            while start < len(self.test_data):
                end = start + batch_size if start + batch_size < len(self.test_data) else len(self.test_data)
                batch_data = dict()
                for field in self.data_fields:
                    batch_data[field] = self.test_data[field][start: end]
                    batch_data_list.append(batch_data)
                start = end
            return batch_data_list

    def load_data(self):
        data_dict = pickle.load(open(self.data_path, "rb"))
        partition_dict = pickle.load(open(self.partition_path, "rb"))
        def generate_client_data(data_dict, index_list):
            data = dict()
            for field in self.data_fields:
                data[field] = [data_dict[field][idx] for idx in index_list]
            return data

        if self.client_idx is None:
            train_index_list = []
            test_index_list = []
            for client_idx in partition_dict[self.partition_method]["partition_data"].keys():
                train_index_list.extend(partition_dict[self.partition_method]["partition_data"][client_idx]["train"])
                test_index_list.extend(partition_dict[self.partition_method]["partition_data"][client_idx]["test"])
            self.train_data = generate_client_data(data_dict, train_index_list)
            self.test_data = generate_client_data(data_dict, test_index_list)
        else:
            train_index_list = partition_dict[self.partition_method]["partition_data"][self.client_idx]["train"]
            test_index_list = partition_dict[self.partition_method]["partition_data"][self.client_idx]["test"]
            self.train_data = generate_client_data(data_dict, train_index_list)
            self.test_data = generate_client_data(data_dict, test_index_list)