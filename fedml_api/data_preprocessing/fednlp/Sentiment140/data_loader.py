import os
import math
import random
import sys
import csv
import time
import datetime


sys.path.append('..')

from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *

class DataLoader(BaseDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "classification"
        self.target_vocab = None

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
            X, Y = self.process_data(self.data_path)
            self.X, self.Y = X, Y
            self.target_vocab = {key: i for i, key in enumerate(set(Y))}
        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type}

    def process_data(self, file_path):
        X = []
        Y = []
        with open(file_path ,"r",newline='',encoding='utf-8',errors='ignore') as csvfile:
            data = csv.reader(csvfile,delimiter=',')
            for line in data:
                X.append(line[5])
                Y.append(line[0])

        return X, Y





if __name__ == "__main__":
    import pickle
    data_path = '../../../../data/fednlp/text_classification/Sentiment140/'
    test_file_path = '../../../../data/fednlp/text_classification/Sentiment140/testdata.manual.2009.06.14.csv'
    train_file_path = '../../../../data/fednlp/text_classification/Sentiment140/training.1600000.processed.noemoticon.csv'
    test_data_loader = DataLoader(test_file_path)

    train_data_loader = DataLoader(train_file_path)
    train_result = train_data_loader.data_loader()

    test_result = test_data_loader.data_loader()

    uniform_partition_dict = uniform_partition([train_result["X"], train_result["Y"]],
                                               [test_result["X"], test_result["Y"]])

    # pickle_dict = train_result
    # pickle_dict["X"].extend(test_result["X"])
    # pickle_dict["Y"].extend(test_result["Y"])
    # pickle.dump(pickle_dict, open("sentiment_140_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict}, open("sentiment_140_partition.pkl", "wb"))

    print("done")
