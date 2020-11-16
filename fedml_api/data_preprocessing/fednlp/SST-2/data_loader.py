import os
import math
import random
from random import shuffle
import sys
import csv
import time


sys.path.append('..')

from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *

class DataLoader(BaseDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "classification"
        self.target_vocab = None
        self.label_file_name = "sentiment_labels.txt"
        self.data_file_name = "dictionary.txt"

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
            X, Y = self.process_data(self.data_path)
            self.X, self.Y = X, Y
            self.target_vocab = {key: i for i, key in enumerate(set(Y))}
        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type}

    def label_level(self, label):
        label = float(label)
        if label >= 0.0 and label <= 0.2:
            return "very negative"
        elif label > 0.2 and label <= 0.4:
            return "negative"
        elif label > 0.4 and label <= 0.6:
            return "neutral"
        elif label > 0.6 and label <= 0.8:
            return "positive"
        else:
            return "very positive"

    def process_data(self, file_path):
        X = []
        Y = []
        label_dict = dict()
        with open(os.path.join(file_path, self.label_file_name)) as f:
            for label_line in f:
                label = label_line.split('|')
                label_dict[label[0].strip()] = label[1]

        with open(os.path.join(file_path, self.data_file_name)) as f:
            for data_line in f:
                data = data_line.strip().split("|")
                X.append(data[0].strip())
                Y.append(self.label_level(label_dict[data[1].strip()]))
        return X, Y


if __name__ == "__main__":
    import pickle
    data_path = '../../../../data//fednlp/text_classification/SST-2/stanfordSentimentTreebank/'

    train_data_loader = DataLoader(data_path)
    train_result = train_data_loader.data_loader()
    uniform_partition_dict = uniform_partition([train_result["X"], train_result["Y"]])
    # pickle.dump(train_result, open("sst_2_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict}, open("sst_2_partition.pkl", "wb"))
    print("done")
