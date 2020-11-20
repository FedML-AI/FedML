import os
import math
import random
import sys
import pickle


sys.path.append('..')

from base.data_loader import BaseRawDataLoader, BaseClientDataLoader
from base.globals import *
from base.partition import *

class RawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "classification"
        self.target_vocab = None
        self.train_file_name = "SemEval2010_task8_training/TRAIN_FILE.txt"
        self.test_file_name = "SemEval2010_task8_testing_keys/TEST_FILE_FULL.txt"

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
            X, Y = self.process_data(os.path.join(self.data_path, self.train_file_name))
            train_size = len(X)
            temp = self.process_data(os.path.join(self.data_path, self.test_file_name))
            X.extend(temp[0])
            Y.extend(temp[1])
            self.X, self.Y = X, Y
            self.target_vocab = {key: i for i, key in enumerate(set(Y))}
            train_index_list = [i for i in range(train_size)]
            test_index_list = [i for i in range(train_size, len(self.X))]
            index_list = train_index_list + test_index_list
            self.attributes["train_index_list"] = train_index_list
            self.attributes["test_index_list"] = test_index_list
            self.attributes["index_list"] = index_list
        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type,
                "attributes": self.attributes}

    def process_data(self, file_path):
        X = []
        Y = []
        with open(file_path, "r", encoding='utf-8') as f:
            data = f.readlines()
            for i in range(len(data)):
                if len(data[i]) > 1 and data[i][0].isdigit():
                    clean_data = data[i].split('\t')[1].strip().replace('"',"")
                    X.append(clean_data)

                elif len(data[i-1]) > 1 and data[i-1][0].isdigit():
                    label = data[i].rstrip("\n")
                    Y.append(label)
        return X, Y

class ClientDataLoader(BaseClientDataLoader):

    def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
        data_fields = ("X", "Y")
        super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
        if self.tokenize:
            self.tokenize_data()

    def tokenize_data(self):
        tokenizer = self.spacy_tokenizer.en_tokenizer

        def __tokenize_data(data):
            for i in range(len(self.data["X"])):
                data["X"][i] = [str(token) for token in tokenizer(data["X"][i])]
                data["Y"][i] = [str(token) for token in tokenizer(data["Y"][i])]

        __tokenize_data(self.train_data)
        __tokenize_data(self.test_data)


# if __name__ == "__main__":
#     data_file_path = '../../../../data//fednlp/text_classification/SemEval2010Task8/SemEval2010_task8_all_data'
#     data_loader = RawDataLoader(data_file_path)
#     results = data_loader.data_loader()
#
#     uniform_partition_dict = uniform_partition(results["attributes"]["train_index_list"], results["attributes"]["test_index_list"])
#
#     pickle.dump(results, open("semeval_2010_task8_data_loader.pkl", "wb"))
#     pickle.dump({"uniform": uniform_partition_dict}, open("semeval_2010_task8_partition.pkl", "wb"))
#     print("done")


